"""Script to extract DINOv2 features from tomograms."""

import logging
import sys
import warnings
from typing import Any

import hydra
from omegaconf import OmegaConf

from cryovit.config import DinoFeaturesConfig
from cryovit.run import dino_features


warnings.simplefilter("ignore")


def validate_config(cfg: DinoFeaturesConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    error_msg = ["The following parameters were missing from dino_features.yaml"]

    for i, key in enumerate(missing_keys, 1):
        param_dict = DinoFeaturesConfig.__annotations__
        error_str = f"{i}. {key}: {param_dict.get(key, Any).__name__}"
        error_msg.append(error_str)

    if missing_keys:
        logging.error("\n".join(error_msg))
        sys.exit(1)


@hydra.main(
    config_path="configs",
    config_name="dino_features.yaml",
    version_base="1.2",
)
def main(cfg: DinoFeaturesConfig) -> None:
    validate_config(cfg)

    try:
        dino_features.process_sample(**cfg)
    except BaseException as err:
        logging.error(f"{type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
