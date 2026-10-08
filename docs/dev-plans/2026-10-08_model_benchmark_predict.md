# Plan de développement — Benchmark multi-modèles en production (predict)

- **Date** : 2026-10-08 (rév. 2 — décisions §11 tranchées)
- **Statut** : ✅ Validé — prêt à implémenter (aucun code écrit)
- **Dépôts impactés** : `geo-deep-learning` (task + docker-compose + configs), `scanfire` (SQL + orchestrateur)
- **Tâche Lightning** : `geo_deep_learning/tasks_with_models/change_detection_changeformer.py` → `ChangeDetectionChangeFormer`

---

## 1. Objectif

Exécuter **plusieurs modèles** (une config YAML de predict par modèle) sur un **jeu de
test de production fixe**, distinct des données d'entraînement, et ranger les sorties
pour comparaison directe :

```
<BENCHMARK_OUTPUT_DIR>/<model_name>/<event_id>/<output_name>.tif
<BENCHMARK_OUTPUT_DIR>/<model_name>/<event_id>/<output_name>_prob.tif
```

- `output_name` = **même convention que le predict de production**
  (`SegmentationOrchestrator.build_output_name`, scanfire) :
  `event-{event_id}_start-{YYYYMMDD}_end_{YYYYMMDD}_pre-g{gid_pre}-{YYYYMMDD}_post-g{gid_post}-{YYYYMMDD}_cell-{cell_id}_beam-{beam}_pass-{sat_pass}.tif`
- Jeu de test défini en base : **1..N groupes PRE** et **1..N groupes ACTIVE/POST** par cas de test.
- Exécution pilotée par un orchestrateur **calqué sur `event_detection_orchestrator`**
  (même image `orchestrator.Dockerfile`, même mount `../scanfire/scanfire_modules`, même
  pattern `Processor` → export CSV → subprocess `train.py predict`), en **passage unique**.

### Hors périmètre (v1)
- Ingestion des géométries dans `event_detection.segmentation_*` (Phase 2 prod) — **interdit**
  pour le benchmark (ne pas polluer les tables de production). Optionnel en v2 (P10, §9).
- Calcul de métriques vs vérité terrain (pas de masque en predict). Optionnel en v2.

---

## 2. Existant réutilisé

| Élément | Emplacement | Réutilisation |
|---|---|---|
| Orchestrateur prod | `scanfire/.../orchestrator/entrypoints/segmentation_orchestrator_service_main.py` | Pattern poll → CSV → predict ; `build_output_name`, `_fmt_date` |
| Base service | `scanfire/.../orchestrator/entrypoints/base.py` (`Processor`) | `run_loop`, `_on_start`, `_poll_and_process` |
| Vue prod | `event_detection.vw_input_files_for_change_detection` | Même schéma de colonnes pour la vue de test |
| Dataset predict | `RCMChangeDetectionOnPredictDataset` | Inchangé — lit `pre_input_file`, `post_input_file`, `pair_id`, `event_id`, `cell_id`, `group_*`, `beam`, `sat_pass`, `event_start_date`, `event_end_date`, `output_name` |
| Écriture GeoTIFF | `ChangeDetectionChangeFormer._write_single_geotiff_prediction` / `on_predict_end` | À paramétrer pour la nouvelle arborescence |
| Nommage / merge | `geo_deep_learning/utils/geotiff_merge.py` (`prediction_output_filename`, `merged_group_filename`, `merge_predictions`) | Inchangé côté nommage |
| Service compose | `geo-deep-learning/docker-compose.yaml` → `event_detection_orchestrator` | Modèle du nouveau service |

### Contraintes actuelles à lever
1. `on_predict_end` force `base_dir = predict_output_dir / "predictions"` si le dossier ne
   s'appelle pas `predictions`.
2. `_write_single_geotiff_prediction` écrit dans `base_dir / event_id / predict_date / cell_id / …`
   → la cible demandée est `base_dir / event_id / …` (pas de `predict_date`, pas de `cell_id`).
3. `merge_predictions` écrit `merged_all.tif` par dossier event/date — mélange de paires
   pre/post différentes, sans intérêt en benchmark (et écrasé à chaque run).
4. `event_end_date` prod = `now()` si NULL → nom de fichier **non déterministe** → casse la
   comparaison inter-modèles. Doit être figé côté vue de test.
5. `manifest.json` unique par `base_dir` → écrasé à chaque run.
6. `event_detection.fct_validate_pre_post_pair()` lit en dur
   `event_detection.event_group_pairs WHERE group_pair_id = NEW.group_pair_id` → réutilisé
   tel quel sur une table de test, il prendrait la géométrie d'une **paire de prod** ayant le
   même ID (collision silencieuse). Refacto rétrocompatible requis (§3.0).

---

## 3. Modèle de données (scanfire, nouveau schéma `model_benchmark`)

Schéma séparé de `deeplearning_dataset` (entraînement) et de `event_detection` (prod) :
aucune écriture dans `event_detection.*`, aucune interaction avec
`vw_input_files_for_change_detection` / `is_predicted`.

**Principe (décision 1)** : la logique applicative reste **définie une seule fois dans
`event_detection`** et est **réutilisée**, pas dupliquée. Les tables `model_benchmark` sont des
**miroirs structurels** de `event_group_pairs` / `event_pre_post_pairs` (mêmes noms de colonnes)
pour que les fonctions trigger `event_detection.*` s'y branchent sans copie.

| Logique `event_detection` | Réutilisation dans `model_benchmark` |
|---|---|
| `events`, `event_cells` (+ `fct_populate_event_cells`) | FK / jointure directe — les cas de test portent sur des événements réels |
| `vw_available_groups_per_event_cell.temporal_position` | Validation du rôle `pre` / `post` d'un groupe (pas de re-codage des dates) |
| `fct_validate_event_group_pair()` | Trigger **tel quel** sur `test_group_pairs` (n'utilise que `NEW.*` + `grouped_frames`) |
| `fct_validate_pre_post_pair()` | Trigger sur `test_pre_post_pairs` **après refacto §3.0** |
| Règle d'appariement de `fct_generate_pre_post_pairs` (beam, sat_pass, relative_orbit avec tolérance `-9999`, `pre_date < post_date`, fichiers fusionnés pre+post, `common_geom` clipé à la cellule) | Reprise dans `fct_generate_test_pairs` — **sans** ranking (`p_max_*_rank`) : tous les pre × tous les post choisis |
| `fct_reset_event_pairs` | Modèle de `fct_reset_test_case_pairs` |
| `vw_input_files_for_change_detection` | Même SQL / mêmes colonnes → `vw_input_files_for_model_test` |
| `fct_set_updated_at()` | Trigger `updated_at` sur `test_cases` |
| GUC `scanfire.skip_pair_trigger` | Réutilisé pour l'insertion en masse |

### 3.0 Refacto rétrocompatible `event_detection.fct_validate_pre_post_pair()`
Lecture de `groups_common_geom` paramétrée par la table parente :
- Argument trigger optionnel `TG_ARGV[0]` = table des paires de groupes (qualifiée).
  Absent ⇒ `event_detection.event_group_pairs` (**comportement prod identique**).
- `EXECUTE format('SELECT groups_common_geom FROM %s WHERE group_pair_id = $1', v_parent_table) USING NEW.group_pair_id`.
- Trigger prod `trg_bi_validate_pre_post_pair` inchangé (sans argument).
- Trigger benchmark : `execute procedure event_detection.fct_validate_pre_post_pair('model_benchmark.test_group_pairs')`.
- Test de non-régression : insertion manuelle dans `event_pre_post_pairs` → même `common_geom` qu'avant.

### 3.1 `model_benchmark.test_cases`
| Colonne | Type | Note |
|---|---|---|
| `test_case_id` | `integer identity PK` | |
| `event_id` | `integer not null FK event_detection.events on delete cascade` | |
| `name` | `varchar(255) not null unique` | Libellé lisible |
| `description` | `text` | |
| `cell_ids` | `varchar[] null` | Restreint aux cellules listées (⊂ `event_detection.event_cells`) ; `NULL` = toutes |
| `reference_end_date` | `date null` | Fige `event_end_date` du nom de fichier ; défaut = `COALESCE(events.end_date, max(post.group_date))` |
| `is_active` | `boolean default true` | |
| `created_at` / `updated_at` | `timestamptz` | trigger `event_detection.fct_set_updated_at()` |

### 3.2 `model_benchmark.test_case_groups` (saisie utilisateur)
| Colonne | Type | Note |
|---|---|---|
| `test_case_id` | `integer FK test_cases on delete cascade` | |
| `group_id` | `integer FK rcm_stac_data.grouped_frames` | |
| `role` | `text check (role in ('pre','post'))` | `post` = `temporal_position` ∈ (`active`,`post`) |
| PK | `(test_case_id, group_id)` | un groupe = un seul rôle par cas |

Trigger `model_benchmark.fct_validate_test_case_group()` (BEFORE INSERT/UPDATE) :
- Le groupe doit apparaître dans `event_detection.vw_available_groups_per_event_cell` pour
  l'`event_id` du cas (⇒ beam A/B, cellule découpée `is_cut`, fenêtre temporelle prod).
- `role='pre'` ⇔ `temporal_position = 'pre'` ; `role='post'` ⇔ `temporal_position IN ('active','post')`. EXCEPTION sinon.
- AFTER INSERT/UPDATE/DELETE : `UPDATE test_cases SET updated_at = now()` (invalide les runs, §3.7).

### 3.3 `model_benchmark.test_group_pairs` (miroir `event_group_pairs`)
Colonnes identiques à `event_detection.event_group_pairs` (`group_pair_id`, `event_id`, `group_id_pre`,
`group_id_post`, `beam`, `sat_pass`, `relative_orbit`, `groups_common_geom`, `is_selected`, `created_at`)
**+ `test_case_id` FK on delete cascade**.
- Unicité `(test_case_id, group_id_pre, group_id_post)`, `check (group_id_pre <> group_id_post)`.
- Trigger : `execute procedure event_detection.fct_validate_event_group_pair()` (réutilisé tel quel).

### 3.4 `model_benchmark.test_pre_post_pairs` (miroir `event_pre_post_pairs`)
Colonnes identiques à `event_detection.event_pre_post_pairs` (`pair_id` identity, `group_pair_id` FK
`test_group_pairs` on delete cascade, `event_id`, `cell_id`, `beam`, `sat_pass`, `group_id_pre`,
`group_id_post`, `is_selected`, `relative_orbit`, `common_geom`, `created_at`) **+ `test_case_id`**.
- **Pas de `is_predicted`** (le suivi est par modèle → `test_runs`).
- Unicité `(test_case_id, cell_id, beam, sat_pass, group_id_pre, group_id_post)`.
- Trigger : `event_detection.fct_validate_pre_post_pair('model_benchmark.test_group_pairs')` (§3.0).
- `pair_id` **stable** (identity) → plus besoin de `row_number()`.

### 3.5 Fonctions
- `model_benchmark.fct_generate_test_pairs(p_test_case_id int) returns int`
  1. Paires de groupes = tous `pre` × tous `post` du cas, même beam / sat_pass, `relative_orbit`
     égal ou `-9999` (règle de `fct_generate_pre_post_pairs`), `pre.group_date < post.group_date`
     → `INSERT … test_group_pairs ON CONFLICT DO NOTHING` (trigger calcule `groups_common_geom`).
  2. Expansion cellules : `event_detection.event_cells` (∩ `cell_ids`) × fichiers fusionnés pre **et**
     post (`grids_scanfire.vw_group_cell_merged_outputs_compat`, `is_merged`) ; `common_geom` clipé
     à `grid_20km` ; `set_config('scanfire.skip_pair_trigger','on',true)` comme en prod.
  3. Idempotent ; `RAISE NOTICE` avec compteurs (paires groupe / cellule, combinaisons rejetées).
- `model_benchmark.fct_reset_test_case_pairs(p_test_case_id int, p_regenerate bool default true)`
  — supprime `test_group_pairs` (cascade) puis régénère. Appelée par l'orchestrateur avant export
  quand `test_cases.updated_at` > dernière génération.

### 3.6 Vues
- `model_benchmark.vw_input_files_for_model_test` — **même SQL que
  `event_detection.vw_input_files_for_change_detection`** sur les tables miroirs, mêmes colonnes
  + `test_case_id`, `test_case_name`, `event_end_date` :
  - `event_end_date = COALESCE(tc.reference_end_date, e.end_date, max(gf_post.group_date) over (partition by tc.test_case_id))` → noms de fichiers déterministes.
  - `WHERE tc.is_active AND pp.is_selected AND gp.is_selected`, beam ∈ (A, B), fichiers pre/post présents.
- `model_benchmark.vw_test_case_summary` — QA : nb groupes pre/post, paires de groupes, paires
  cellule, cellules sans fichier, par cas.

### 3.7 `model_benchmark.test_runs` (suivi, équivalent `is_predicted`)
| Colonne | Type |
|---|---|
| `run_id` | `integer identity PK` |
| `test_case_id` | `FK test_cases` |
| `model_name` | `varchar not null` |
| `config_path`, `checkpoint_path`, `config_sha256` | `text` |
| `status` | `text check in ('running','succeeded','failed')` |
| `n_pairs` | `integer` |
| `output_dir` | `text` |
| `started_at`, `finished_at` | `timestamptz` |
| `error` | `text` |

Cas « à traiter » pour un modèle = actif **et** (aucun run `succeeded` pour `(test_case_id, model_name, config_sha256)` **ou** `test_cases.updated_at > finished_at`).

Fichiers SQL à créer (convention `scanfire/database_schemas/<schema>/`) :
- `model_benchmark/schema.sql`
- `test_cases.sql`, `test_case_groups.sql`, `test_group_pairs.sql`, `test_pre_post_pairs.sql`, `test_runs.sql` (+ séquences/index/triggers dans chaque fichier, comme `event_detection`)
- `fct_validate_test_case_group.sql`, `fct_touch_test_case.sql`, `fct_generate_test_pairs.sql`, `fct_reset_test_case_pairs.sql`
- `vw_input_files_for_model_test.sql`, `vw_test_case_summary.sql`
- `install.sql`, `seed_example.sql`, `tests/test_model_benchmark.sql` (détail et ordre : P2, §9)
- Modifié : `event_detection/fct_validate_pre_post_pair.sql` (§3.0)

Droits : `owner to scanfire_maintainer`, `grant select/insert/update/delete … to scanfire_user`
sur `test_cases` / `test_case_groups`, `grant execute` sur les fonctions.

Exemple de saisie :
```sql
INSERT INTO model_benchmark.test_cases (event_id, name) VALUES (42, 'feu_42_juin');
INSERT INTO model_benchmark.test_case_groups (test_case_id, group_id, role) VALUES
  (1, 1701, 'pre'), (1, 1702, 'pre'),
  (1, 2301, 'post'), (1, 2350, 'post'), (1, 2399, 'post');
SELECT model_benchmark.fct_generate_test_pairs(1);   -- optionnel : l'orchestrateur l'appelle
SELECT * FROM model_benchmark.vw_test_case_summary;  -- QA
```

---

## 4. Registre des modèles (geo-deep-learning)

Nouveau fichier `configs/benchmark/models.yaml` :
```yaml
output_root: /mnt/geospatial/projet_RCM_scanfire/benchmark_outputs
models:
  - name: cs2base_cosine_restarts_e24          # → dossier <output_root>/<name>/
    config: /app/configs/benchmark/cs2base_cosine_restarts_predict.yaml
    enabled: true
  - name: changeformer_v7_all9bands
    config: /app/configs/benchmark/changeformer_v7_all9bands_predict.yaml
    enabled: true
```

- Une **config predict par modèle** (dérivée de `configs/rcm_change_detection_predict.yaml`) :
  `change_detection_model`, flags (`use_metadata_film`, `use_cbam`, `use_signed_difference`, …),
  `band_names`, `separate_metadata`, `weights_from_checkpoint_path` **doivent matcher l'entraînement**.
- `name` validé : `^[A-Za-z0-9_.-]+$`, unique.
- Les configs `fit` ne sont **pas** utilisables telles quelles (dataset train, callbacks, devices) —
  documenter la procédure de dérivation dans `configs/benchmark/README.md`.

### Override injecté par l'orchestrateur
LightningCLI accepte plusieurs `--config` (le dernier gagne). Pour chaque (modèle, run) l'orchestrateur écrit
`benchmark_override.yaml` :
```yaml
trainer:
  devices: 1
  callbacks: []
  logger: false
model:
  init_args:
    predict_output_dir: <output_root>/<model_name>
    predict_output_layout: benchmark
    predict_run_name: <model_name>
    predict_write_merged_all: false
data:
  init_args:
    dataset_class: datasets.rcm_change_detection_predict_dataset.RCMChangeDetectionOnPredictDataset
    csv_root_folder: <BENCHMARK_CSV_DIR>/<model_name>
    csv_file_name: vw_input_files_for_model_test.csv
```
Commande : `python /app/geo_deep_learning/train.py predict --config <model.yaml> --config <override.yaml>`.

Filtres `beams` (décision 5) : **conservés** depuis la config du modèle — l'override ne touche
jamais `beams` / `satellite_pass` / `dataset_years` (un modèle entraîné beam A seulement ne doit
pas recevoir du B). L'orchestrateur logge le nb de paires exportées ; le dataset logge le nb
retenu après filtre → écart visible dans les logs et dans `test_runs.n_pairs`.

---

## 5. Modifications `ChangeDetectionChangeFormer` (geo-deep-learning)

Nouveaux `init_args` (défauts = comportement prod **inchangé**) :

| Arg | Type / défaut | Effet |
|---|---|---|
| `predict_output_layout` | `Literal["production","benchmark"] = "production"` | Arborescence de sortie |
| `predict_run_name` | `str \| None = None` | Ajouté au manifest (`run_name`) |
| `predict_write_merged_all` | `bool = True` | Désactive `merged_all.tif` |

Comportement `benchmark` :
- `on_predict_end` : `base_dir = Path(predict_output_dir)` **sans** suffixe `/predictions`.
- Nouveau helper `_prediction_dir(base_dir, event_id, predict_date, cell_id) -> Path` :
  - `production` → `base_dir / event_id / predict_date / cell_id` (actuel)
  - `benchmark` → `base_dir / event_id`
- `_write_single_geotiff_prediction` (aujourd'hui `@staticmethod`) : reçoit le dossier cible
  (ou `layout`) au lieu de le calculer ; clé `event_date_key` = dossier event en benchmark.
- Sorties conservées (décision 2) — **les deux** :
  - TIF **par cellule** : `<model>/<event_id>/event-…_cell-<id>_beam-…_pass-….tif` (+ `_prob.tif`) ;
  - TIF **fusionné par paire pre/post** (`merged_group_filename`, sans cellule) :
    `<model>/<event_id>/event-…_pre-g…_post-g…_beam-A_pass-ASC.tif`.
  Collision impossible (seuls les fichiers par cellule contiennent `_cell-`).
- `merge_predictions(..., write_merged_all=False)` : paramètre ajouté dans `utils/geotiff_merge.py`.
- Manifest : `base_dir / "_runs" / predict_date / "manifest.json"` + copie `base_dir / "manifest_latest.json"`.
  Chemins TIF absolus dans le manifest → toujours exploitables.
- Tolérance à l'écrasement : re-run du même modèle ⇒ fichiers écrasés (idempotent, voulu).

Points de vigilance :
- Ne **pas** toucher `NO_DATA=32767`, `PROBABILITY_NODATA=-1.0`, TTA, crop du buffer (`_crop_prediction_to_cell`).
- Les deux chemins d'écriture (per-tile `_write_prediction_batch` et blending `_save_assembled_predictions`)
  passent par `_write_single_geotiff_prediction` → un seul point à modifier.
- `rcm_change_detection_predict.yaml` prod : aucune modification requise.

---

## 6. Orchestrateur benchmark (scanfire)

### 6.1 Code
- Extraire `_fmt_date` / `build_output_name` dans `scanfire_modules/pipeline/orchestrator/prediction_naming.py` ;
  `SegmentationOrchestrator` les ré-exporte (rétrocompat, aucune modif de comportement).
- Nouveau `entrypoints/model_benchmark_orchestrator_service_main.py` → `ModelBenchmarkOrchestrator(Processor)`.
  **Un seul passage puis arrêt** (décision 4) : `main()` appelle `_on_start()` + `_poll_and_process()`
  une fois, code retour ≠ 0 si au moins un modèle a échoué. Pas de boucle, pas de polling.
  ```
  _poll_and_process():
      models = load_registry(BENCHMARK_MODELS_FILE)          # enabled only
      for tc in active test cases (filtre --test-case):
          if tc.updated_at > last generation: fct_reset_test_case_pairs(tc)   # §3.5
      for model in models:
          df = query_pending(model)                         # §3.7 + vw_input_files_for_model_test
          if df.empty: continue
          csv = export_csv(df, model)                       # + output_name (prediction_naming)
          ovr = write_override(model)                       # §4
          run_id(s) = insert test_runs(status='running')
          rc = subprocess.run([... predict --config model.config --config ovr])
          update test_runs(status=succeeded|failed, finished_at, n_pairs, error)
          snapshot: copier model.config + ovr + csv → <output_root>/<name>/_runs/<ts>/
  ```
  - Échec d'un modèle ⇒ log + `failed`, **on continue** avec le modèle suivant.
  - Modèles exécutés **séquentiellement** (`devices: 1`).
  - Aucune Phase 2 (ingestion) : pas d'écriture dans `event_detection.*`.
- Shim `orchestrator/model_benchmark_orchestrator.py` (même pattern que `segmentation_orchestrator.py`).
- CLI : `--models-file`, `--model NAME` (filtre, répétable), `--test-case ID` (filtre), `--force` (ignore `test_runs`), `--dry-run` (export CSV + override seulement).

### 6.2 Variables d'environnement
| Variable | Défaut | Description |
|---|---|---|
| `DATABASE_URL` | — | Même DB que scanfire |
| `BENCHMARK_MODELS_FILE` | `/app/configs/benchmark/models.yaml` | Registre des modèles |
| `BENCHMARK_OUTPUT_DIR` | valeur `output_root` du registre | Racine des sorties (surcharge) |
| `BENCHMARK_CSV_DIR` | `/mnt/geospatial/projet_RCM_scanfire/benchmark_outputs/_csv` | CSV d'entrée par modèle |
| `LOGGER_LEVEL` | `INFO` | |

---

## 7. Docker / Makefile (geo-deep-learning)

Nouveau service dans `docker-compose.yaml`, profil `benchmark` (ne démarre pas avec `make up`) :
```yaml
  model_benchmark_orchestrator:
    build:
      context: .
      dockerfile: orchestrator.Dockerfile
    runtime: nvidia
    shm_size: '32gb'
    container_name: model_benchmark_orchestrator
    profiles: ["benchmark"]
    restart: "no"
    user: '9005'
    env_file: [.env]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTORCH_ALLOC_CONF=expandable_segments:True
      - LOGGER_LEVEL=INFO
      - BENCHMARK_MODELS_FILE=/app/configs/benchmark/models.yaml
    volumes:
      - "./configs:/app/configs:ro"
      - "./models_checkpoints:/app/models_checkpoints:ro"
      - "../scanfire/scanfire_modules:/app/scanfire_modules:ro"
      - "/mnt/geospatial/projet_RCM_scanfire:/mnt/geospatial/projet_RCM_scanfire:rw"
    entrypoint: ["python", "/app/scanfire_modules/pipeline/orchestrator/model_benchmark_orchestrator.py"]
    deploy:
      <<: *gpu-deploy
```
GPU partagé avec `event_detection_orchestrator` (décision 3) : aucune gestion spécifique,
même `deploy` GPU que la prod. `restart: "no"` → le conteneur s'arrête après le passage unique.

Makefile : `benchmark-start` (`$(COMPOSE) --profile benchmark run --rm model_benchmark_orchestrator`),
`benchmark-dry-run` (idem + `--dry-run`), `logs-benchmark`.

---

## 8. Tests

### geo-deep-learning (`tests/`)
- `_prediction_dir` : layout `production` vs `benchmark`.
- `on_predict_end` benchmark (prédictions factices, `tmp_path`) : arborescence `<out>/<event_id>/<output_name>.tif` + `_prob.tif`, pas de `predictions/`, pas de `predict_date`, pas de `merged_all.tif`, manifest dans `_runs/`.
- Non-régression layout `production` (chemins identiques à aujourd'hui).
- `merge_predictions(write_merged_all=False)`.

### scanfire (`tests/unit/`)
- `prediction_naming.build_output_name` : identique à l'ancienne implémentation (golden strings).
- Chargement / validation registre (`name` invalide, doublon, config absente).
- Génération override YAML.
- `ModelBenchmarkOrchestrator` avec engine + `subprocess.run` mockés : échec modèle 1 n'empêche pas modèle 2 ; statuts `test_runs` ; un seul passage puis sortie (code retour).
- SQL (`database_schemas/model_benchmark/tests/test_model_benchmark.sql`, manuel, en transaction + `ROLLBACK`) :
  - groupe `pre` daté après `start_date` → rejeté par `fct_validate_test_case_group` ;
  - pre/post beam ou sat_pass différents → non appariés par `fct_generate_test_pairs` ;
  - insertion manuelle dans `test_pre_post_pairs` → `common_geom` calculé depuis `test_group_pairs` (pas depuis la prod) ;
  - **non-régression** `event_detection.fct_validate_pre_post_pair()` sans argument (prod) ;
  - `vw_input_files_for_model_test` : mêmes colonnes que `vw_input_files_for_change_detection` (+ `test_case_id`, `test_case_name`, `event_end_date`).

---

## 9. Phases de développement

### 9.0 Vue d'ensemble

| Phase | Titre | Dépôt | Dépend de | Effort | Parallélisable avec |
|---|---|---|---|---|---|
| **P0** | Préparation & cadrage | les deux | — | S | — |
| **P1** | Refacto rétrocompatible `fct_validate_pre_post_pair` | scanfire (SQL) | P0 | S | P3, P4 |
| **P2** | Schéma `model_benchmark` (tables, fonctions, vues) | scanfire (SQL) | P1 | L | P3, P4, P5 |
| **P3** | Extraction `prediction_naming` | scanfire (Python) | P0 | S | P1, P2, P4 |
| **P4** | Layout `benchmark` dans `ChangeDetectionChangeFormer` | geo-deep-learning | P0 | M | P1, P2, P3 |
| **P5** | Registre modèles + configs predict par modèle | geo-deep-learning | P4 | M | P2 |
| **P6** | `ModelBenchmarkOrchestrator` | scanfire (Python) | P2, P3 | L | P5 |
| **P7** | Intégration Docker / Makefile | geo-deep-learning | P5, P6 | S | — |
| **P8** | Recette bout-en-bout | les deux | P7 | M | — |
| **P9** | Documentation & clôture | les deux | P8 | S | — |
| *P10* | *(v2, optionnel)* Analyse comparative inter-modèles | scanfire | P9 | L | — |

Effort : S ≤ ½ j · M ≈ 1–2 j · L ≈ 2–4 j.

```
P0 ─┬─ P1 ── P2 ──┐
    ├─ P3 ────────┼── P6 ──┐
    └─ P4 ── P5 ──┴────────┴── P7 ── P8 ── P9 ── (P10)
```

Branches : `feature/model-benchmark` dans **chaque** dépôt. Un commit (ou une PR) par phase.
Règle transverse : **à chaque phase, la prod reste fonctionnelle** (`event_detection_orchestrator`
et `rcm_change_detection_predict.yaml` inchangés en comportement).

---

### P0 — Préparation & cadrage
**Objectif** : réunir les données d'entrée du benchmark avant tout code.

Tâches :
- [ ] Créer les branches `feature/model-benchmark` (geo-deep-learning à partir de la branche `xm/changeformer-add-segformer`, scanfire).
- [ ] Lister les modèles à comparer : `name`, config d'entraînement d'origine, checkpoint `.ckpt`, `band_names`, flags (FiLM, CBAM, signed-diff, …), `beams`.
- [ ] Choisir 1–2 événements de test (`event_detection.events`) et leurs groupes pre / active-post
      (`SELECT … FROM event_detection.vw_available_groups_per_event_cell WHERE event_id = …`).
- [ ] Vérifier que ces événements **ne sont pas** dans le jeu d'entraînement
      (`deeplearning_dataset.tbl_input_pre_post_data` / `split_contents_*.csv`, colonne `db_nbac_fire_id`).
- [ ] Définir `BENCHMARK_OUTPUT_DIR` sur `/mnt/geospatial/projet_RCM_scanfire/…` + droits UID 9005.
- [ ] Identifier une base de dev / staging pour tester le SQL (ou procédure transaction + `ROLLBACK`).

**Livrable** : tableau modèles + liste des cas de test ajoutés en annexe de ce plan (§12).
**Critère de sortie** : chaque modèle a un checkpoint accessible depuis le conteneur (`/app/models_checkpoints/…`).

---

### P1 — Refacto rétrocompatible `event_detection.fct_validate_pre_post_pair()`
**Objectif** : rendre le trigger réutilisable par `model_benchmark` sans changer la prod (§3.0).

Fichiers :
- `scanfire/database_schemas/event_detection/fct_validate_pre_post_pair.sql` (modifié)
- `scanfire/database_schemas/model_benchmark/tests/test_fct_validate_pre_post_pair_regression.sql` (nouveau)

Tâches :
- [ ] Lire la table parente via `TG_NARGS` / `TG_ARGV[0]` ; défaut `event_detection.event_group_pairs`.
- [ ] Remplacer le `SELECT … INTO` par `EXECUTE format('… FROM %s …', v_parent_table) USING NEW.group_pair_id` (nom validé par `to_regclass`, sinon `RAISE EXCEPTION`).
- [ ] Conserver le fast-path `scanfire.skip_pair_trigger` et le fallback `ST_Intersection`.
- [ ] Mettre à jour le `comment on function`.
- [ ] Test de non-régression (transaction + `ROLLBACK`) : insertion dans `event_pre_post_pairs` avec et sans `group_pair_id` → `beam`, `sat_pass`, `common_geom` identiques à l'ancienne version.

**Critère de sortie** : trigger prod `trg_bi_validate_pre_post_pair` inchangé (pas d'argument), test de non-régression OK, déployé sur staging.

---

### P2 — Schéma `model_benchmark`
**Objectif** : modèle de données de test, réutilisant la logique `event_detection` (§3.1 → §3.7).

Fichiers (`scanfire/database_schemas/model_benchmark/`) :

| Ordre d'exécution | Fichier | Contenu |
|---|---|---|
| 1 | `schema.sql` | `create schema model_benchmark`, owner, grants usage |
| 2 | `test_cases.sql` | table + trigger `event_detection.fct_set_updated_at()` |
| 3 | `fct_validate_test_case_group.sql` | validation rôle via `vw_available_groups_per_event_cell` |
| 4 | `fct_touch_test_case.sql` | AFTER I/U/D → `test_cases.updated_at = now()` |
| 5 | `test_case_groups.sql` | table + 2 triggers ci-dessus |
| 6 | `test_group_pairs.sql` | miroir `event_group_pairs` + trigger `event_detection.fct_validate_event_group_pair()` |
| 7 | `test_pre_post_pairs.sql` | miroir `event_pre_post_pairs` + trigger `event_detection.fct_validate_pre_post_pair('model_benchmark.test_group_pairs')` |
| 8 | `test_runs.sql` | suivi par (cas, modèle, `config_sha256`) |
| 9 | `fct_generate_test_pairs.sql` | expansion pre × post → paires groupe → paires cellule |
| 10 | `fct_reset_test_case_pairs.sql` | suppression cascade + régénération |
| 11 | `vw_input_files_for_model_test.sql` | même SQL/colonnes que la vue prod + `test_case_id`, `test_case_name`, `event_end_date` |
| 12 | `vw_test_case_summary.sql` | QA par cas |
| 13 | `install.sql` | `\i` des fichiers 1→12 dans l'ordre |
| 14 | `tests/test_model_benchmark.sql` | scénarios §8 (transaction + `ROLLBACK`) |
| 15 | `seed_example.sql` | cas de test de P0 |

Tâches clés :
- [ ] `fct_generate_test_pairs` : reprendre **à l'identique** la jointure prod (beam, sat_pass, `relative_orbit` avec tolérance `-9999`, `pre_date < post_date`, `vw_group_cell_merged_outputs_compat.is_merged`, clip `grid_20km`), sans ranking ; filtrer `cell_ids`.
- [ ] `vw_input_files_for_model_test` : `event_end_date` déterministe (`COALESCE(reference_end_date, end_date, max(post.group_date))`).
- [ ] Ajouter `test_cases.pairs_generated_at` (ou équivalent) pour savoir quand régénérer.
- [ ] Grants : `scanfire_user` en lecture/écriture sur `test_cases` / `test_case_groups`, exécution des fonctions.

**Critère de sortie** : `install.sql` idempotent sur staging ; `tests/test_model_benchmark.sql` vert ;
`seed_example.sql` → `vw_input_files_for_model_test` non vide avec des chemins de fichiers existants.

---

### P3 — Extraction `prediction_naming` (scanfire)
**Objectif** : une seule implémentation de la convention de nommage, partagée prod / benchmark.

Fichiers :
- `scanfire_modules/pipeline/orchestrator/prediction_naming.py` (nouveau) : `fmt_date()`, `build_output_name(row)`, `add_output_names(df, default_end_date)`.
- `entrypoints/segmentation_orchestrator_service_main.py` (modifié) : `_fmt_date` / `build_output_name` délèguent au module (signatures conservées).
- `tests/unit/test_prediction_naming.py` (nouveau).

Tâches :
- [ ] Golden tests : sorties **identiques** à l'implémentation actuelle (dates `None`, `NaN`, `str`, `date`, `Timestamp`).
- [ ] `add_output_names` : la prod garde `default_end_date = today` ; le benchmark passe `None` (date déjà figée par la vue).

**Critère de sortie** : tests verts, `ruff check .` OK, aucun changement de nom de fichier en prod.

---

### P4 — Layout `benchmark` dans `ChangeDetectionChangeFormer` (geo-deep-learning)
**Objectif** : écrire `<out>/<model_name>/<event_id>/<output_name>.tif` sans impacter la prod (§5).

Fichiers :
- `geo_deep_learning/tasks_with_models/change_detection_changeformer.py`
- `geo_deep_learning/utils/geotiff_merge.py`
- `tests/test_change_detection_predict_output_layout.py` (nouveau)

Tâches :
- [ ] Ajouter `predict_output_layout` (`"production"` | `"benchmark"`, validé dans `__init__` → `ValueError`), `predict_run_name`, `predict_write_merged_all`.
- [ ] Helper `_resolve_predict_base_dir()` (suffixe `/predictions` seulement en `production`).
- [ ] Helper `_prediction_dir(base_dir, event_id, predict_date, cell_id)`.
- [ ] `_write_single_geotiff_prediction` : recevoir `target_dir` + `merge_dir` au lieu de les calculer (les 2 appelants : `_write_prediction_batch`, `_save_assembled_predictions`).
- [ ] `merge_predictions(..., write_merged_all: bool = True)`.
- [ ] Manifest : `production` → inchangé ; `benchmark` → `_runs/<predict_date>/manifest.json` + `manifest_latest.json`, champ `run_name`, `output_layout`.
- [ ] Tests (prédictions factices + `tmp_path`, sans GPU) :
  - `benchmark` : TIF par cellule + `_prob.tif` + TIF fusionné par paire dans `<out>/<event_id>/`, pas de `merged_all.tif`, pas de `predictions/`, pas de dossier date/cellule ;
  - `production` : arborescence identique à aujourd'hui (non-régression) ;
  - les deux chemins d'écriture (per-tile et overlap-blended).

**Critère de sortie** : tests verts ; `get_errors` propre ; un predict prod réel produit la même arborescence qu'avant.

---

### P5 — Registre modèles + configs predict par modèle (geo-deep-learning)
**Objectif** : une config predict valide par modèle, référencée dans un registre (§4).

Fichiers :
- `configs/benchmark/models.yaml`
- `configs/benchmark/<model_name>_predict.yaml` (un par modèle de P0)
- `configs/benchmark/README.md` (procédure : config `fit` → config predict ; champs à recopier ; pièges `in_channels`, `separate_metadata`, `use_signed_difference`)
- `scripts/validate_benchmark_configs.py` (nouveau)

Tâches :
- [ ] Dériver chaque config de `rcm_change_detection_predict.yaml` + hyperparamètres de la config d'entraînement du checkpoint.
- [ ] `validate_benchmark_configs.py` : pour chaque modèle du registre, `train.py predict --config <cfg> --print_config` (parsing LightningCLI), existence du checkpoint, unicité/format de `name`, cohérence `in_channels == len(band_names) + 1`.
- [ ] Optionnel : instancier le modèle + charger les poids sur CPU (`strict=True`) pour détecter les incohérences d'architecture tôt.

**Critère de sortie** : script de validation vert pour tous les modèles du registre.

---

### P6 — `ModelBenchmarkOrchestrator` (scanfire)
**Objectif** : passage unique : préparer les paires → exporter → predict par modèle → tracer (§6).

Fichiers (`scanfire_modules/pipeline/orchestrator/`) :
- `benchmark/registry.py` — chargement / validation `models.yaml` (dataclass `BenchmarkModel`).
- `benchmark/repository.py` — accès DB : cas actifs, `fct_reset_test_case_pairs`, paires à traiter, CRUD `test_runs`.
- `benchmark/override.py` — génération `benchmark_override.yaml` + `config_sha256`.
- `entrypoints/model_benchmark_orchestrator_service_main.py` — `ModelBenchmarkOrchestrator(Processor)` + `main()`.
- `model_benchmark_orchestrator.py` — shim (même pattern que `segmentation_orchestrator.py`).
- `entrypoints/base.py` — ajouter `ModelBenchmarkOrchestrator` dans la docstring de hiérarchie.
- `tests/unit/test_model_benchmark_registry.py`, `test_model_benchmark_override.py`, `test_model_benchmark_orchestrator.py`.

Tâches :
- [ ] Régénération des paires si `test_cases.updated_at > pairs_generated_at`.
- [ ] Sélection « à traiter » par modèle (§3.7) ; filtres CLI `--model`, `--test-case`, `--force`, `--dry-run`.
- [ ] Export CSV `<BENCHMARK_CSV_DIR>/<model_name>/vw_input_files_for_model_test.csv` via `prediction_naming.add_output_names`.
- [ ] `subprocess.run([... "predict", "--config", cfg, "--config", ovr])` ; capture rc ; `test_runs` `running` → `succeeded`/`failed` (+ `error` = fin du stderr).
- [ ] Snapshot `<output_root>/<model_name>/_runs/<ts>/` (config, override, CSV).
- [ ] Échec d'un modèle → on continue ; `main()` retourne `1` si au moins un échec.
- [ ] Tests : engine + `subprocess.run` mockés ; échec modèle 1 / succès modèle 2 ; `--dry-run` sans subprocess ; aucun appel SQL vers `event_detection` en écriture.

**Critère de sortie** : tests verts, `ruff check .` OK, `--dry-run` sur staging produit CSV + override corrects.

---

### P7 — Intégration Docker / Makefile (geo-deep-learning)
**Objectif** : lancement en une commande, même logique que `event_detection_orchestrator` (§7).

Fichiers : `docker-compose.yaml`, `Makefile`, `.env` (doc des variables, pas de secrets commités).

Tâches :
- [ ] Service `model_benchmark_orchestrator` (profil `benchmark`, `restart: "no"`, image `orchestrator.Dockerfile`, mounts `configs`, `models_checkpoints`, `../scanfire/scanfire_modules`, `/mnt/geospatial/…`).
- [ ] Cibles `benchmark-start`, `benchmark-dry-run`, `logs-benchmark` + entrées dans `help`.
- [ ] Vérifier que `make up` / `orchestrator-start` ne démarrent **pas** le benchmark.

**Critère de sortie** : `docker compose --profile benchmark config` valide ; `make benchmark-dry-run` OK.

---

### P8 — Recette bout-en-bout
**Objectif** : valider la chaîne complète sur données réelles.

Scénarios :
1. [ ] `make benchmark-dry-run` → CSV + override par modèle, aucun predict, aucun `test_runs` `succeeded`.
2. [ ] 1 cas × 1 modèle → arborescence `<OUT>/<model>/<event_id>/` conforme (par cellule + `_prob` + fusionné par paire), `test_runs` `succeeded`, conteneur arrêté.
3. [ ] 1 cas × N modèles → listings identiques entre modèles (`diff` des noms de fichiers vide).
4. [ ] Relance sans modification → aucun modèle relancé (« à traiter » vide).
5. [ ] Ajout d'un groupe post au cas → régénération des paires + relance de tous les modèles.
6. [ ] Modèle avec checkpoint invalide → `failed`, les autres `succeeded`, code retour 1.
7. [ ] Modèle beam A uniquement sur cas A+B → seules les paires A prédites (filtre conservé).
8. [ ] Pendant la recette, `event_detection_orchestrator` continue de tourner normalement (GPU partagé).
9. [ ] Contrôle visuel QGIS : alignement géographique, nodata `32767` / `-1.0`, buffer rogné.

**Critère de sortie** : critères §10 tous cochés.

---

### P9 — Documentation & clôture
Tâches :
- [ ] `scanfire/AGENTS.md` + `README.md` : section « Model benchmark » (tables, fonctions, service, variables).
- [ ] `scanfire/docs/architecture*.md` : ajout du schéma `model_benchmark` et du service.
- [ ] `geo-deep-learning/configs/benchmark/README.md` finalisé ; nouveaux `init_args` documentés dans la docstring de `ChangeDetectionChangeFormer`.
- [ ] Ce plan → statut « ✅ Réalisé », écarts éventuels notés.
- [ ] Merge des branches `feature/model-benchmark` (scanfire **avant** geo-deep-learning : le compose monte `../scanfire/scanfire_modules`).

---

### P10 — *(v2, optionnel)* Analyse comparative inter-modèles
- Vectorisation des `_prob.tif` par zones (`probability_thresholds`) via la logique existante de
  `segment_ingestion.py`, vers `model_benchmark.results` (pas `event_detection.segmentation_*`).
- Vues de comparaison : surface brûlée par seuil / modèle / paire, IoU croisé entre modèles,
  comparaison NBAC si disponible.
- Rapport (notebook ou script) généré à partir de ces vues.

---

## 10. Critères d'acceptation

- [ ] Ajout d'un cas de test = `INSERT` SQL (`test_cases` + `test_case_groups`), aucun code modifié.
- [ ] `make benchmark-start` produit, pour chaque modèle activé : `<OUT>/<model_name>/<event_id>/event-…_cell-…_beam-…_pass-….tif` (+ `_prob.tif`) **et** le TIF fusionné par paire pre/post, puis le conteneur s'arrête.
- [ ] Noms de fichiers **identiques entre modèles** pour un même cas (diff de listing vide).
- [ ] Prod inchangée : `rcm_change_detection_predict.yaml` + `event_detection_orchestrator` produisent exactement la même arborescence qu'avant ; trigger prod `fct_validate_pre_post_pair` même résultat.
- [ ] Aucune écriture dans `event_detection.*` par le benchmark.
- [ ] Aucune logique métier dupliquée : validation des groupes / paires via les fonctions et vues `event_detection`.
- [ ] Échec d'un modèle tracé dans `test_runs`, les autres continuent.

## 11. Décisions (tranchées le 2026-10-08)

| # | Question | Décision |
|---|---|---|
| 1 | Schéma | Nouveau schéma `model_benchmark`, **réutilisant la logique `event_detection`** (tables miroirs + triggers/vues/fonctions `event_detection`, cf. §3) |
| 2 | Sorties | **Garder les 2** : TIF par cellule **et** TIF fusionné par paire pre/post (`merged_all.tif` désactivé) |
| 3 | GPU | Partagé avec `event_detection_orchestrator`, aucune gestion spécifique |
| 4 | Exécution | **Un seul passage puis arrêt** (pas de polling, `restart: "no"`) |
| 5 | Filtres beam | **Conservés** depuis la config de chaque modèle (jamais surchargés) |

## 12. Annexe — Inventaire (à compléter en P0)

### 12.1 Modèles
| `name` (registre) | Config d'entraînement | Checkpoint | `change_detection_model` | `band_names` | `beams` | Flags (FiLM / CBAM / signed-diff / DFA) |
|---|---|---|---|---|---|---|
| | | | | | | |

### 12.2 Cas de test
| `name` | `event_id` | Groupes pre (`group_id` / date) | Groupes active-post (`group_id` / date) | `cell_ids` | `reference_end_date` | Absent du jeu d'entraînement ? |
|---|---|---|---|---|---|---|
| | | | | | | |