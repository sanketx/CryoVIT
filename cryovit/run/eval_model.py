import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import Callback
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

from cryovit.config import EvalModelConfig
from cryovit.config import ExpPaths
from cryovit.models.base_model import BaseModel


torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class TestPredictionWriter(Callback):
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
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
    dataset_type = HydraConfig.get().runtime.choices.dataset

    match dataset_type:
        case "single" | "loo" | "fractional":
            exp_paths.exp_dir /= cfg.dataset.sample.name

        case "multi":
            samples = sorted([s.name for s in cfg.dataset.sample])
            exp_paths.exp_dir /= "_".join(samples)

    match dataset_type:
        case "multi" | "loo" | "fractional":
            split_ids = [cfg.dataset.split_id]

        case "single":
            split_ids = range(10)

    for split_id in split_ids:
        split_dir = "" if split_id is None else f"split_{split_id}"
        ckpt_dir = exp_paths.exp_dir / split_dir
        ckpt_path = ckpt_dir / "weights.pt"

        if not ckpt_dir.exists():
            raise ValueError(f"The directory {ckpt_dir} does not exist")

        if not ckpt_path.exists():
            raise ValueError(f"{ckpt_dir} does not contain a checkpoint")


def build_datamodule(cfg: EvalModelConfig) -> LightningDataModule:
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


def get_scores(
    model: BaseModel,
    ckpt_dir: Path,
    result_dir: Path,
    cfg: EvalModelConfig,
) -> pd.DataFrame:
    state_dict = torch.load(ckpt_dir / "weights.pt")
    model.load_state_dict(state_dict)
    datamodule = build_datamodule(cfg)

    trainer = instantiate(cfg.trainer)
    test_writer = TestPredictionWriter(result_dir)
    trainer.callbacks.append(test_writer)

    trainer.test(model, datamodule)
    return test_writer.result_df


def run_trainer(cfg: EvalModelConfig):
    dataset_type = HydraConfig.get().runtime.choices.dataset
    exp_paths = cfg.exp_paths
    validate_paths(exp_paths, cfg)

    split_id = cfg.dataset.split_id
    split_dir = "" if split_id is None else f"split_{split_id}"
    model = instantiate(cfg.model)

    match dataset_type:
        case "single":
            results = []
            result_dir = exp_paths.exp_dir.parent / "results"

            for i in range(10):
                cfg.dataset.split_id = i
                ckpt_dir = exp_paths.exp_dir / f"split_{i}"
                result_df = get_scores(model, ckpt_dir, result_dir, cfg)
                results.append(result_df)

            result_df = pd.concat(results, axis=0, ignore_index=True)
            result_file = result_dir / f"{exp_paths.exp_dir.name}.csv"

        case "multi":
            ckpt_dir = exp_paths.exp_dir / split_dir
            result_dir = exp_paths.exp_dir / "results" / split_dir
            result_df = get_scores(model, ckpt_dir, result_dir, cfg)

            test_samples = sorted([s.name for s in cfg.dataset.test_samples])
            result_file = result_dir / f"{'_'.join(test_samples)}.csv"

        case "loo" | "fractional":
            ckpt_dir = exp_paths.exp_dir / split_dir
            result_dir = exp_paths.exp_dir.parent / "results" / split_dir
            result_df = get_scores(model, ckpt_dir, result_dir, cfg)
            result_file = result_dir / f"{exp_paths.exp_dir.name}.csv"

    result_df.to_csv(result_file, index=False)
    print(result_df.describe())
