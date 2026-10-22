"""Scheduler builders that cannot be expressed purely in the YAML config.

Used by the change-detection task's ``configure_optimizers``:

* :func:`with_warmup` — prepend a linear warm-up (``SequentialLR``), which
  jsonargparse cannot instantiate from YAML since nested schedulers need the
  optimizer.
* :func:`build_onecycle_scheduler` — ``OneCycleLR`` whose ``total_steps`` is
  inferred from the Lightning trainer / DataModule.
"""

import math
from typing import Any

import torch

ONECYCLE_EXTRA_KEYS = (
    "pct_start", "anneal_strategy", "div_factor",
    "final_div_factor", "three_phase", "cycle_momentum",
)


def scheduler_class_path(spec: Any) -> str | None:
    """Dotted class path of a scheduler spec.

    ``hparams["scheduler"]`` is a ``{"class_path": ...}`` dict when built by
    LightningCLI, but the raw callable when the task is instantiated in code
    (tests, notebooks) — handle both instead of assuming the dict.
    """
    if isinstance(spec, dict):
        return spec.get("class_path")
    func = getattr(spec, "func", spec)  # functools.partial
    if isinstance(func, type):
        return f"{func.__module__}.{func.__qualname__}"
    return None


def with_warmup(
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        warmup_iters: int,
        start_factor: float,
) -> Any:
    """Prepend a linear LR warm-up to ``scheduler`` (no-op when ``warmup_iters <= 0``)."""
    if warmup_iters <= 0:
        return scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        msg = "lr_warmup_iters is not supported with ReduceLROnPlateau (not chainable)"
        raise ValueError(msg)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, scheduler],
        milestones=[warmup_iters],
    )


def build_onecycle_scheduler(
        optimizer: torch.optim.Optimizer,
        trainer: Any,
        init_args: dict[str, Any],
) -> torch.optim.lr_scheduler.OneCycleLR:
    """Build a ``OneCycleLR`` scheduler, inferring ``total_steps`` when needed.

    Lightning's ``trainer.estimated_stepping_batches`` is normally used,
    but falls back to computing steps from the DataModule's
    ``epoch_size`` (IterableDataset case) or, failing that, to an
    explicit ``total_steps`` from the YAML config.
    """
    max_lr = init_args.get("max_lr")
    extra_kwargs = {key: init_args[key] for key in ONECYCLE_EXTRA_KEYS if key in init_args}

    stepping_batches = trainer.estimated_stepping_batches
    if stepping_batches > -1:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=stepping_batches,
            **extra_kwargs,
        )

    epoch_size = getattr(trainer.datamodule, "epoch_size", None)
    if stepping_batches == -1 and epoch_size is not None:
        batch_size = trainer.datamodule.batch_size
        accumulate_grad_batches = trainer.accumulate_grad_batches
        max_epochs = trainer.max_epochs
        steps_per_epoch = math.ceil(
            epoch_size / (batch_size * accumulate_grad_batches),
        )
        buffer_steps = int(steps_per_epoch * accumulate_grad_batches)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch + buffer_steps,
            epochs=max_epochs,
            **extra_kwargs,
        )

    total_steps = init_args.get("total_steps")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        **extra_kwargs,
    )
