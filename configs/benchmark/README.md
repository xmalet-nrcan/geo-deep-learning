# Benchmark predict multi-modèles — configs

Plan : [`docs/dev-plans/2026-10-08_model_benchmark_predict.md`](../../docs/dev-plans/2026-10-08_model_benchmark_predict.md).

Exécute **plusieurs modèles** sur un jeu de test de production fixe
(`model_benchmark.vw_input_files_for_model_test`, scanfire) et range les sorties
pour comparaison :

```
<output_root>/<name>/<event_id>/<output_name>.tif          # par cellule (classes, nodata 32767)
<output_root>/<name>/<event_id>/<output_name>_prob.tif     # probabilité brûlé (float32, nodata -1.0)
<output_root>/<name>/<event_id>/<fusion par paire pre/post>.tif   # sans _cell-
<output_root>/<name>/_runs/<YYYYMMDD_HHMM>/manifest.json   # historique
<output_root>/<name>/manifest_latest.json
```

`output_name` = convention prod (`scanfire …/orchestrator/prediction_naming.py`) →
noms **identiques entre modèles** pour un même cas de test. Pas de `merged_all.tif`.

## Fichiers

| Fichier | Rôle |
|---|---|
| `models.yaml` | Registre : `output_root` + `models[]` (`name`, `config`, `enabled`, `description`) |
| `<name>_predict.yaml` | Config predict d'un modèle (une par entrée du registre) |
| `_template_predict.yaml` | Modèle de départ (non référencé) |
| `../../scripts/validate_benchmark_configs.py` | Validation registre + configs + checkpoints |

Règles du registre : `name` ∈ `^[A-Za-z0-9_.-]+$`, unique (= dossier de sortie et
`model_benchmark.test_runs.model_name`) ; `config` relatif = relatif à `models.yaml`.

## Dériver une config predict depuis une config d'entraînement

Les configs `fit` ne sont **pas** utilisables telles quelles (dataset train, callbacks,
logger, devices, split). Partir de `_template_predict.yaml` (ou d'une config existante
ici) et recopier depuis la config d'entraînement **du checkpoint** :

| Bloc | Champs à recopier à l'identique | Pourquoi |
|---|---|---|
| `model.init_args` | `change_detection_model`, `num_classes`, `use_metadata_film`, `film_embed_dim`, `use_cbam`, `cbam_reduction`, `use_dfa`, `dfa_gate_hidden`, `use_signed_difference`, `signed_difference_channels`, `signed_difference_normalize`, `backbone_kwargs` | Architecture : sinon couches absentes/aléatoires |
| `model.init_args` | `image_size` | Padding des tuiles |
| `data.init_args` | `band_names` (**même ordre**), `bands`, `separate_metadata` | Canaux d'entrée |
| `data.init_args` | `tile_size`, `tile_stride`, `predict_overlap_buffer` (= `train_overlap_buffer`), `patch_size` | Contexte spatial vu à l'entraînement |
| `data.init_args` | `beams`, `satellite_pass`, `dataset_years` | Filtres **conservés** (jamais surchargés) : un modèle beam A ne reçoit pas de B |
| `trainer` | `precision` | Cohérence numérique |
| `model.init_args` | `weights_from_checkpoint_path` | Chemin **conteneur** (`/app/models_checkpoints/…`) |

Pièges :
- **`in_channels`** : ne pas le renseigner — inféré de `band_names`
  (`len(band_names) + 1` pour `BITMASK_CROPPED`, `+3` si `separate_metadata: false`).
  S'il est renseigné et faux, le datamodule l'emporte (warning) → masque l'erreur.
- **`separate_metadata` / `use_metadata_film`** : `use_metadata_film: true` exige
  `separate_metadata: true` (SAT_PASS/BEAM en scalaires FiLM, pas en bandes).
- **`use_signed_difference`** : ajoute des canaux **internes** à l'encodeur ; ne change
  pas `in_channels` déclaré. Doit matcher l'entraînement (sinon `size mismatch` du stem).
- **Chargement strict en benchmark** : la prod charge en `strict=False` (incohérence
  d'archi = simple *warning* + couches aléatoires). Le benchmark utilise les meilleurs
  poids de chaque modèle (même archi) → l'override force `weights_strict: true` : toute
  clé manquante/inattendue ou `size mismatch` fait **échouer** le run (`test_runs.status
  = failed`). `validate_benchmark_configs.py --load-weights` le détecte en amont.
  `weights_strict: true` est incompatible avec `load_parts`.
- Les losses / optimizer / scheduler sont requis par le constructeur mais inutilisés.

## Override injecté par l'orchestrateur

Ne **pas** renseigner ces clés dans `<name>_predict.yaml` (warning du validateur) ;
`csv_root_folder` / `csv_file_name` restent à `__set_by_benchmark_override__`
(obligatoires pour le parseur, remplacés à l'exécution).

```yaml
trainer:   {devices: 1, callbacks: [], logger: false}
model:
  init_args:
    predict_output_dir: <output_root>/<name>
    predict_output_layout: benchmark
    predict_run_name: <name>
    predict_write_merged_all: false
    weights_strict: true
data:
  init_args:
    dataset_class: datasets.rcm_change_detection_predict_dataset.RCMChangeDetectionOnPredictDataset
    csv_root_folder: <BENCHMARK_CSV_DIR>/<name>
    csv_file_name: vw_input_files_for_model_test.csv
```

Commande exécutée par modèle :
`python /app/geo_deep_learning/train.py predict --config <name>_predict.yaml --config benchmark_override.yaml`.
Référence : `build_override()` dans `scripts/validate_benchmark_configs.py`
(à garder synchronisé avec `scanfire …/orchestrator/benchmark/override.py`, P6).

## Validation

```bash
# Registre + configs + hparams des checkpoints (dans le conteneur orchestrateur)
python scripts/validate_benchmark_configs.py
# + parsing LightningCLI avec l'override + chargement strict des poids sur CPU
python scripts/validate_benchmark_configs.py --cli --load-weights
# Un modèle (même désactivé), poste de dev avec checkpoints locaux
python scripts/validate_benchmark_configs.py --model <name> --cli \
    --path-map /app/models_checkpoints=D:/models_checkpoints
```

| Contrôle | Option | Détecte |
|---|---|---|
| Registre | toujours | `name` invalide/doublon, `config` absente, `enabled` non booléen |
| Config statique | toujours | mauvaise tâche, `band_names` absent, FiLM sans `separate_metadata`, `in_channels` incohérent, clés surchargées |
| Checkpoint | défaut (`--skip-checkpoint`) | fichier absent ; hparams d'archi ≠ config ; `in_channels` du checkpoint ≠ `band_names` |
| LightningCLI | `--cli` | erreur de parsing de `predict --config … --config override` |
| Poids | `--load-weights` | clés manquantes / inattendues, `size mismatch` |

Code retour `1` si au moins une erreur. `--path-map SRC=DST` : découpage sur le
**dernier** `=` (les noms de checkpoints contiennent `epoch=…`).

## Ajouter un modèle

1. Copier `_template_predict.yaml` → `<name>_predict.yaml`, compléter les `TODO`.
2. Ajouter l'entrée dans `models.yaml` (`enabled: true`).
3. `python scripts/validate_benchmark_configs.py --model <name> --cli --load-weights` → `OK`.
4. Lancer le benchmark (P7 : `make benchmark-dry-run` puis `make benchmark-start`).
