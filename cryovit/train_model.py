"""Script to train segmentation models for CryoET data."""

import logging
import sys
import warnings
from typing import Any

import hydra
from omegaconf import OmegaConf

from cryovit.config import TrainModelConfig
from cryovit.run import train_model


warnings.simplefilter("ignore")


def validate_config(cfg: TrainModelConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    error_msg = ["The following parameters were missing from train_model.yaml"]

    for i, key in enumerate(missing_keys, 1):
        error_msg.append(f"{i}. {key}")

    if missing_keys:
        logging.error("\n".join(error_msg))
        sys.exit(1)


@hydra.main(
    config_path="configs",
    config_name="train_model.yaml",
    version_base="1.2",
)
def main(cfg: TrainModelConfig) -> None:
    validate_config(cfg)

    try:
        train_model.run_trainer(cfg)
    except BaseException as err:
        logging.error(f"{type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
