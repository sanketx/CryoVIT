import os

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule

from cryovit.config import TrainModelConfig
from cryovit.data_modules import MultiSampleDataModule


torch.set_float32_matmul_precision("high")


def build_datamodule(cfg: TrainModelConfig) -> LightningDataModule:
    samples = [s.name for s in cfg.experiment.samples]
    split_id = cfg.experiment.split_id
    split_file = cfg.experiment.split_file
    dataloader_fn = instantiate(cfg.dataloader)

    if cfg.model._target_ == "cryovit.models.CryoVIT":
        input_key = "dino_features"
    else:
        input_key = "data"

    dataset_params = {
        "input_key": input_key,
        "label_key": cfg.experiment.label_key,
        "data_root": cfg.experiment.tomo_dir,
    }

    return MultiSampleDataModule(
        samples,
        split_id,
        split_file=split_file,
        dataloader_fn=dataloader_fn,
        dataset_params=dataset_params,
    )


def run_trainer(cfg: TrainModelConfig):
    exp_dir = cfg.experiment.exp_dir / cfg.experiment.exp_name
    os.makedirs(exp_dir, exist_ok=True)

    datamodule = build_datamodule(cfg)
    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    trainer.fit(model, datamodule)
