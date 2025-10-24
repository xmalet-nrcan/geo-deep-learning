"""MLFlow logger."""

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger

def safe_name(name: str) -> str:
    """Sanitize names for MLflow (artifact paths, run IDs, etc.)."""
    return (
        str(name)
        .replace("=", "-")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "-")
    )

class LoggerSaveConfigCallback(SaveConfigCallback):
    """Save configuration file as an MLflow artifact safely."""

    def save_config(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
        """Save config to MLflow as an artifact."""
        if isinstance(trainer.logger, MLFlowLogger):
            try:
                # Récupération du chemin local du fichier de config
                config_filepath = self.config.config[0]
                print(f"[MLflow] Saving config: {config_filepath}")
                print(safe_name(trainer.logger.run_id))
                # Log de l'artifact avec noms nettoyés
                trainer.logger.experiment.log_artifact(
                    local_path=config_filepath,
                    artifact_path=safe_name("config"),
                    run_id=safe_name(trainer.logger.run_id),
                )

                print("[MLflow] Config file successfully logged as artifact.")

            except Exception as e:
                print(f"[MLflow] Failed to log config artifact: {e}")
