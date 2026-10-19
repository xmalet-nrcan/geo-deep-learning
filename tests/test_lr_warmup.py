"""Tests for the linear LR warm-up built in ``configure_optimizers``."""

import functools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("kornia")
smp_losses = pytest.importorskip("segmentation_models_pytorch.losses")

from geo_deep_learning.tasks_with_models.change_detection_changeformer import (  # noqa: E402
    ChangeDetectionChangeFormer,
)
from geo_deep_learning.tools.schedulers.builders import scheduler_class_path  # noqa: E402


def _task(scheduler, **kwargs) -> ChangeDetectionChangeFormer:
    task = ChangeDetectionChangeFormer(
        "changestar2",
        image_size=(32, 32),
        num_classes=1,
        max_samples=0,
        main_loss=smp_losses.FocalLoss(mode="binary"),
        secondary_loss=smp_losses.LovaszLoss(mode="binary"),
        in_channels=3,
        use_metadata_film=False,
        optimizer=functools.partial(torch.optim.AdamW, lr=3e-4),
        scheduler=scheduler,
        **kwargs,
    )
    task.configure_model()
    return task


def _lr_trajectory(task: ChangeDetectionChangeFormer, n: int) -> list[float]:
    (optimizer,), (config,) = task.configure_optimizers()
    lrs = []
    for _ in range(n):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        config["scheduler"].step()
    return lrs


CAWR = functools.partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=20, eta_min=1e-6)


def test_warmup_then_cosine_restarts() -> None:
    lrs = _lr_trajectory(_task(CAWR, lr_warmup_iters=5, lr_warmup_start_factor=0.1), 46)
    assert lrs[0] == pytest.approx(3e-5)
    assert lrs[0] < lrs[2] < lrs[4] < lrs[5]  # linear ramp
    assert lrs[5] == pytest.approx(3e-4)  # warm-up done -> cosine starts at max
    assert lrs[24] < 1e-5  # end of first cycle
    assert lrs[25] == pytest.approx(3e-4)  # restart
    assert lrs[45] == pytest.approx(3e-4)


def test_no_warmup_keeps_plain_scheduler() -> None:
    (_, ), (config,) = _task(CAWR).configure_optimizers()
    assert isinstance(config["scheduler"], torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


def test_scheduler_class_path_from_code() -> None:
    path = scheduler_class_path(_task(CAWR).hparams.get("scheduler"))
    assert path == "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"


def test_invalid_warmup_arguments() -> None:
    with pytest.raises(ValueError, match="lr_warmup_iters"):
        _task(CAWR, lr_warmup_iters=-1)
    with pytest.raises(ValueError, match="lr_warmup_start_factor"):
        _task(CAWR, lr_warmup_iters=5, lr_warmup_start_factor=0.0)
