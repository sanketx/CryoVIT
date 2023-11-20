"""Config file for CryoVIT experiments."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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


cs = ConfigStore.instance()
cs.store(name="dino_features_config", node=DinoFeaturesConfig)

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
