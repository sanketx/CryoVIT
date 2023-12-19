import logging
import os

import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from pytorch_lightning import seed_everything

from cryovit.config import ExpPaths
from cryovit.config import Sample
from cryovit.config import TrainModelConfig


seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


def set_wandb_config(cfg: TrainModelConfig):
    if isinstance(cfg.dataset.sample, Sample):
        sample = cfg.dataset.sample.name
    else:
        sample = [s.name for s in cfg.dataset.sample]

    config = {
        "model": cfg.model._target_.split(".")[-1],
        "experiment": cfg.exp_name,
        "split_id": cfg.dataset.split_id,
        "sample": sample,
    }

    for logger in cfg.trainer.logger:
        if logger._target_.split(".")[-1] == "WandbLogger":
            logger.config.update(config)


def setup_params(exp_paths: ExpPaths, cfg: TrainModelConfig):
    dataset_type = HydraConfig.get().runtime.choices.dataset

    match dataset_type:
        case "single" | "loo" | "fractional":
            exp_paths.exp_dir /= cfg.dataset.sample.name

        case "multi":
            samples = sorted([s.name for s in cfg.dataset.sample])
            exp_paths.exp_dir /= "_".join(samples)

    match dataset_type:
        case "single" | "multi" | "loo":
            split_id = cfg.dataset.split_id
            split_dir = "" if split_id is None else f"split_{split_id}"
            exp_paths.exp_dir /= split_dir

        case "fractional":
            split_id = cfg.dataset.split_id
            exp_paths.exp_dir /= f"split_{split_id}"

            if not 1 <= split_id <= 10:
                raise ValueError(f"split_id: {split_id} must be between 1 and 10")

    os.makedirs(exp_paths.exp_dir, exist_ok=True)


def build_datamodule(cfg: TrainModelConfig) -> LightningDataModule:
    model_type = HydraConfig.get().runtime.choices.model

    match model_type:
        case "cryovit":
            input_key = "dino_features"
        case _:
            input_key = "data"

    dataset_params = {
        "input_key": input_key,
        "label_key": cfg.label_key,
        "data_root": cfg.exp_paths.tomo_dir,
        "aux_keys": cfg.aux_keys,
    }

    return instantiate(cfg.dataset)(
        split_file=cfg.exp_paths.split_file,
        dataloader_fn=instantiate(cfg.dataloader),
        dataset_params=dataset_params,
    )


def run_trainer(cfg: TrainModelConfig):
    exp_paths = cfg.exp_paths
    setup_params(exp_paths, cfg)
    set_wandb_config(cfg)

    datamodule = build_datamodule(cfg)
    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)

    trainer.fit(model, datamodule)
    torch.save(model.state_dict(), (exp_paths.exp_dir / "weights.pt"))
    wandb.finish()
