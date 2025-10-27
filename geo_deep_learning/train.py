"""Train model with Lightning CLI."""

import logging

from lightning.pytorch import seed_everything
from lightning.pytorch.cli import ArgsType, LightningCLI

from configs import logging_config  # noqa: F401
from geo_deep_learning.tools.mlflow_logger import LoggerSaveConfigCallback


def safe_name(name: str) -> str:
    """Replace invalid MLflow characters in artifact or run names."""
    return (
        name.replace("=", "-")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "-")
    )


logger = logging.getLogger(__name__)


def main(args: ArgsType = None) -> None:
    """Run the main training pipeline."""
    seed_everything(42, workers=True)
    cli = LightningCLI(
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        auto_configure_optimizers=False,
        args=args,
    )


    if cli.trainer.is_global_zero:
        logger.info(
            "Best model path: %s",
            cli.trainer.checkpoint_callback.best_model_path,
        )
        cli.trainer.logger.log_hyperparams(
            {"best_model_path": cli.trainer.checkpoint_callback.best_model_path},
        )
        logger.info("Done!")


if __name__ == "__main__":
    main()
