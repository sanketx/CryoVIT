import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from tqdm import tqdm

from .. import config
from ..data_modules import MultiSampleDataModule
from ..datasets import DinoDataset
from ..models import BasicUpsample
from ..models.losses import DiceLoss
from ..models.metrics import DiceMetric


torch.set_float32_matmul_precision("high")
exp_dir = os.path.join(config.EXP_DIR, "dino")

dataloader_params = {
    "batch_size": None,
    "pin_memory": True,
    "num_workers": 8,
    "prefetch_factor": 1,
    "persistent_workers": False,
}

datamodule_params = {
    "train_samples": ["WT"],
    "test_samples": ["dN17_BACHD"],
    "split_id": 3,
    "split_type": "split_10",
    "split_file": os.path.join(config.EXP_DIR, "splits.csv"),
    "dataset_class": defaultdict(lambda: DinoDataset),
    "dataset_params": {},
    "dataloader_params": dataloader_params,
}

model_params = {
    "losses": [(DiceLoss, {})],
    "metrics": [(DiceMetric, {"threshold": 0.5})],
    "lr": 1e-4,
}

monitor = "VAL_DiceMetric"
mode = "max"

model_checkpoint = ModelCheckpoint(
    dirpath=os.path.join(exp_dir, "checkpoints"),
    filename="dino",
    monitor=monitor,
    mode=mode,
    # save_top_k=-1,
    # save_on_train_epoch_end=True,
    # save_weights_only=True,
    every_n_epochs=1,
)

early_stopping = EarlyStopping(
    monitor=monitor,
    mode=mode,
    check_on_train_epoch_end=True,
    min_delta=0.01,
    patience=200,
    stopping_threshold=0.985,
)

swa = StochasticWeightAveraging(
    swa_lrs=1e-2,
    swa_epoch_start=40,
)

csv_logger = CSVLogger(
    save_dir=os.path.join(exp_dir, "csv_logs"),
    name="",
    flush_logs_every_n_steps=1,
)

wandb_logger = WandbLogger(
    save_dir=exp_dir,
    project="cryovit",
    mode="disabled",
)

trainer_params = dict(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    log_every_n_steps=1,
    num_sanity_val_steps=0,
    accumulate_grad_batches=4,
    logger=[csv_logger, wandb_logger],
    callbacks=[model_checkpoint, early_stopping, swa, RichProgressBar()],
    max_epochs=-1,
    # gradient_clip_val=0.5,
    # gradient_clip_algorithm="value",
    default_root_dir="/tmp/sanketg",
)


def evaluate(model, datamodule, ckpt_name):
    results = defaultdict(list)
    trainer = Trainer(accelerator="gpu", precision="16-mixed", devices=1)
    preds = trainer.predict(
        model,
        dataloaders=datamodule.test_dataloader(),
        ckpt_path=os.path.join(exp_dir, "checkpoints", ckpt_name),
    )

    for p in tqdm(preds):
        sample = p["sample"]
        tomo_name = p["tomo_name"]
        data = p["data"]
        mito_pred = p["mito_pred"]
        mito_true = p["mito_true"]
        results["tomo_name"].append(tomo_name)

        sample_dir = os.path.join(config.ROOT, "debug", sample)
        os.makedirs(sample_dir, exist_ok=True)

        with h5py.File(os.path.join(sample_dir, tomo_name), "w") as fh:
            fh.create_dataset("data", data=np.squeeze(data.numpy()))
            fh.create_dataset("pred", data=np.squeeze(mito_pred.numpy()))
            fh.create_dataset(
                "mito",
                data=np.squeeze(mito_true.numpy()).astype(np.int8),
                compression="gzip",
            )

        mito_weight = (mito_true > -1.0).view(-1, 1)
        mito_pred = torch.masked_select(mito_pred.view(-1, 1), mito_weight)
        mito_true = torch.masked_select(mito_true.view(-1, 1), mito_weight)

        for loss_fn in model.loss_fns["TEST"]:
            loss_val = loss_fn(mito_pred, mito_true).cpu().numpy()
            results[f"TEST_{type(loss_fn).__name__}"].append(loss_val)

        for metric_fn in model.metric_fns["TEST"]:
            metric_fn(mito_pred, mito_true)
            metric_val = metric_fn.compute().cpu().numpy()
            metric_fn.reset()
            results[f"TEST_{type(metric_fn).__name__}"].append(metric_val)

    result_df = pd.DataFrame.from_dict(results)
    result_file = os.path.join(config.ROOT, "debug", "results.csv")
    result_df.to_csv(result_file, index=False)

    result_df = pd.read_csv(result_file)
    print(result_df.describe())


if __name__ == "__main__":
    datamodule = MultiSampleDataModule(**datamodule_params)
    model = BasicUpsample(**model_params)
    # trainer = Trainer(**trainer_params)

    # trainer.fit(
    #     model,
    #     train_dataloaders=datamodule.train_dataloader(),
    #     val_dataloaders=datamodule.val_dataloader(),
    #     # ckpt_path=os.path.join(exp_dir, "checkpoints", "ckpt_name"),
    # )

    evaluate(model, datamodule, "dino-v112.ckpt")
