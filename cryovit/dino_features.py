"""Script to extract DINOv2 features from tomograms."""

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cryovit.config import Sample
from cryovit.data_proc import dino_features


@dataclass
class DinoConfig:
    dino_dir: Path
    data_dir: Path
    feature_dir: Path
    batch_size: int
    sample: Sample


cs = ConfigStore.instance()
cs.store(name="base_config", node=DinoConfig)
warnings.simplefilter("ignore")


def validate_config(cfg: DinoConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    error_msg = ["The following parameters were missing from dino_features.yaml"]

    for i, key in enumerate(missing_keys, 1):
        error_str = f"{i}. {key}: {DinoConfig.__annotations__.get(key, Any).__name__}"
        error_msg.append(error_str)

    if missing_keys:
        logging.error("\n".join(error_msg))
        sys.exit(1)


@hydra.main(
    config_path="configs",
    config_name="dino_features.yaml",
    version_base="1.2",
)
def main(cfg: DinoConfig) -> None:
    validate_config(cfg)

    try:
        dino_features.process_sample(**cfg)
    except BaseException as err:
        logging.error(f"{type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
