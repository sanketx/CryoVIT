"""Script to train segmentation models for CryoET data."""

import logging
import sys
import warnings

import hydra
from omegaconf import OmegaConf

from cryovit.config import TrainModelConfig
from cryovit.run import train_model


warnings.simplefilter("ignore")


def validate_config(cfg: TrainModelConfig) -> None:
    """Validates the configuration for training segmentation models on CryoET data.

    Checks if all necessary parameters are present in the configuration. Logs an error and exits if any
    required parameters are missing.

    Args:
        cfg (TrainModelConfig): The configuration object containing settings for model training.

    Raises:
        SystemExit: If any configuration parameters are missing, terminating the script.
    """
    missing_keys = OmegaConf.missing_keys(cfg)
    error_msg = ["The following parameters were missing from train_model.yaml"]

    for i, key in enumerate(missing_keys, 1):
        error_msg.append(f"{i}. {key}")

    if missing_keys:
        logging.error("\n".join(error_msg))
        sys.exit(1)

    OmegaConf.set_struct(cfg, False)


@hydra.main(
    config_path="configs",
    config_name="train_model.yaml",
    version_base="1.2",
)
def main(cfg: TrainModelConfig) -> None:
    """Main function to orchestrate the training of segmentation models.

    Validates the provided configuration, then initializes and runs the training process using the
    specified settings. Catches and logs any exceptions that occur during training.

    Args:
        cfg (TrainModelConfig): Configuration object loaded from train_model.yaml.

    Raises:
        BaseException: Captures and logs any exceptions that occur during the training process.
    """
    validate_config(cfg)

    try:
        train_model.run_trainer(cfg)
    except BaseException as err:
        logging.error(f"{type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
