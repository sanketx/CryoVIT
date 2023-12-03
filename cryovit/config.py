"""Config file for CryoVIT experiments."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from hydra.core.config_store import ConfigStore


samples = [
    "BACHD",
    "BACHD_controlRNA",
    "BACHD_pias1",
    "dN17_BACHD",
    "Q109",
    "Q18",
    "Q20",
    "Q53",
    "Q53_KD",
    "Q66",
    "Q66_GRFS1",
    "Q66_KD",
    "WT",
    "cancer",
]

Sample = Enum("Sample", samples)


@dataclass
class DinoFeaturesConfig:
    dino_dir: Path
    data_dir: Path
    feature_dir: Path
    batch_size: int
    sample: Sample
    all_samples: str = ",".join(samples)
    cryovit_root: Optional[Path] = None


@dataclass
class Experiment:
    exp_name: str
    label_key: str
    samples: List[Sample]
    split_id: int

    exp_dir: Path
    tomo_dir: Path
    split_file: Path
    cryovit_root: Optional[Path] = None


@dataclass
class BaseModel:
    lr: float
    weight_decay: float = 1e-3
    losses: Tuple[Dict] = (dict(_target_="cryovit.models.losses.DiceLoss"),)
    metrics: Tuple[Dict] = (
        dict(_target_="cryovit.models.metrics.DiceMetric", threshold=0.5),
    )


@dataclass
class CryoVIT(BaseModel):
    _target_: str = "cryovit.models.CryoVIT"


@dataclass
class DataLoader:
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool = False
    pin_memory: bool = True
    batch_size: Optional[int] = None
    _target_: str = "torch.utils.data.DataLoader"
    _partial_: bool = True


@dataclass
class Trainer:
    accelerator: str = "gpu"
    devices: str = 1
    precision: str = "16-mixed"
    _target_: str = "pytorch_lightning.Trainer"


@dataclass
class TrainerFit(Trainer):
    logger: Optional[List] = None
    callbacks: Optional[List] = None

    max_epochs: int = 50
    accumulate_grad_batches: int = 4

    log_every_n_steps: int = 1
    num_sanity_val_steps: int = 0


@dataclass
class TrainModelConfig:
    model: BaseModel
    trainer: Trainer
    dataloader: DataLoader
    experiment: Experiment


cs = ConfigStore.instance()
cs.store(name="dino_features_config", node=DinoFeaturesConfig)

cs.store(group="model", name="cryovit", node=CryoVIT)
cs.store(group="trainer", name="trainer_fit", node=TrainerFit)
cs.store(name="train_model_config", node=TrainModelConfig)


UNET_PATCH_SIZE = (128, 512, 512)  # size of patch fed into the model
DINO_PATCH_SIZE = (128, 32, 32)

# Path to original high res Huntington tomograms
HD_DIR = "/sdf/home/s/sanketg/C036/20210811_mitochondria_for_joy/for_huntinton_paper"

# Project root directory
ROOT = "/sdf/home/s/sanketg/projects/CryoVIT"

# Path to updated raw data - float 32 with proper distribution alignment
RAW_TOMO_DIR = f"{ROOT}/data/tomograms/raw"

# Path to csv with annotation information
ANNOTATION_CSV = f"{ROOT}/data/csv"

# Path to mitochondria and granule masks
HASTY_ANNOTATIONS = f"{ROOT}/data/images/annotations"

# Path to training data with new hasty.ai annotations
TRAIN_TOMO_DIR = f"{ROOT}/data/tomograms/train"

# Path to datasets splits
EXP_DIR = f"{ROOT}/experiments"

# Path to DinoV2 foundation models
DINO_DIR = "/sdf/home/s/sanketg/projects/foundation_models"

# Path to DinoV2 training data
DINO_TOMO_DIR = f"{ROOT}/data/tomograms/dino"
# DINO_TOMO_DIR = "/tmp/sanketg/dino"
AI_TOMO_DIR = f"{ROOT}/ai_labels"
