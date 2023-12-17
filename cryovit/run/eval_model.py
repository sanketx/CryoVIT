import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Optional

import h5py
import pandas as pd
import torch
from hydra.utils import instantiate
from pytorch_lightning import Callback
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

from cryovit.config import EvalModelConfig
from cryovit.config import ExpPaths
from cryovit.data_modules import MultiSampleDataModule

from ..models.base_model import BaseModel


torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class TestPredictionWriter(Callback):
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir / "results"
        self.scores = defaultdict(list)

    def _save_prediction(self, outputs):
        tomo_dir = self.results_dir / outputs["sample"]
        os.makedirs(tomo_dir, exist_ok=True)

        with h5py.File(tomo_dir / outputs["tomo_name"], "w") as fh:
            fh.create_dataset("data", data=outputs["data"])
            fh.create_dataset("preds", data=outputs["preds"], compression="gzip")
            fh.create_dataset("label", data=outputs["label"], compression="gzip")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._save_prediction(outputs)

        for key, value in outputs.items():
            if key not in ("data", "label", "preds"):
                self.scores[key].append(value)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.result_df = pd.DataFrame.from_dict(self.scores)


def validate_paths(exp_paths: ExpPaths, cfg: EvalModelConfig):
    exp_paths.exp_dir /= cfg.exp_name
    split_ids = [cfg.split_id] if cfg.split_id is not None else range(10)

    for split_id in split_ids:
        exp_dir = exp_paths.exp_dir / f"split_{split_id}"
        ckpt_path = exp_dir / "weights.pt"

        if not exp_dir.exists():
            raise ValueError(f"The directory {exp_dir} does not exist")

        if not ckpt_path.exists():
            raise ValueError(f"{ckpt_path.parent} does not contain a checkpoint")


def build_datamodule(
    cfg: EvalModelConfig, split_id: Optional[int]
) -> LightningDataModule:
    samples = [s.name for s in cfg.samples]
    split_file = cfg.exp_paths.split_file
    dataloader_fn = instantiate(cfg.dataloader)

    if cfg.model._target_ == "cryovit.models.CryoVIT":
        input_key = "dino_features"
    else:
        input_key = "data"

    dataset_params = {
        "input_key": input_key,
        "label_key": cfg.label_key,
        "data_root": cfg.exp_paths.tomo_dir,
        "aux_keys": ["data"],
    }

    return MultiSampleDataModule(
        samples,
        split_id,
        split_file=split_file,
        dataloader_fn=dataloader_fn,
        dataset_params=dataset_params,
    )


def get_scores(
    model: BaseModel,
    model_split_id: int,
    target_split_id: Optional[int],
    exp_paths: ExpPaths,
    cfg: EvalModelConfig,
) -> pd.DataFrame:
    exp_dir = exp_paths.exp_dir / f"split_{model_split_id}"
    state_dict = torch.load(exp_dir / "weights.pt")
    model.load_state_dict(state_dict)
    datamodule = build_datamodule(cfg, target_split_id)

    trainer = instantiate(cfg.trainer)
    test_writer = TestPredictionWriter(exp_dir.parent.parent)
    trainer.callbacks.append(test_writer)

    trainer.test(model, datamodule)
    return test_writer.result_df


def run_trainer(cfg: EvalModelConfig):
    exp_paths = cfg.exp_paths
    validate_paths(exp_paths, cfg)
    model = instantiate(cfg.model)

    result_file = exp_paths.exp_dir.parent / "results"
    results = []

    if cfg.split_id is not None:
        result_df = get_scores(model, cfg.split_id, None, exp_paths, cfg)
        results.append(result_df)

    else:
        for split_id in range(10):
            result_df = get_scores(model, split_id, split_id, exp_paths, cfg)
            results.append(result_df)

    result_df = pd.concat(results, axis=0, ignore_index=True)
    sample = f"{cfg.samples[0].name}_" if len(cfg.samples) == 1 else ""
    split_id = f"_{cfg.split_id}" if cfg.split_id is not None else ""
    result_file /= f"{sample}results{split_id}.csv"

    result_df.to_csv(result_file, index=False)
    print(result_df.describe())
