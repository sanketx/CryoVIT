"""Script to evaluate segmentation models for CryoET data."""

import logging
import sys
import warnings
from typing import Any

import hydra
from omegaconf import OmegaConf

from cryovit.config import EvalModelConfig
from cryovit.run import eval_model


warnings.simplefilter("ignore")


def validate_config(cfg: EvalModelConfig) -> None:
    """Validates the configuration for evaluating segmentation models on CryoET data.

    Ensures that all necessary parameters are included in the configuration file. If any parameters are
    missing, it logs an error message and exits the script to prevent further execution.

    Args:
        cfg (EvalModelConfig): The configuration object containing settings for model evaluation.

    Raises:
        SystemExit: If any configuration parameters are missing, thereby terminating the script.
    """
    missing_keys = OmegaConf.missing_keys(cfg)
    error_msg = ["The following parameters were missing from eval_model.yaml"]

    for i, key in enumerate(missing_keys, 1):
        error_msg.append(f"{i}. {key}")

    if missing_keys:
        logging.error("\n".join(error_msg))
        sys.exit(1)


@hydra.main(
    config_path="configs",
    config_name="eval_model.yaml",
    version_base="1.2",
)
def main(cfg: EvalModelConfig) -> None:
    """Main function to orchestrate the evaluation of segmentation models.

    First, it validates the configuration and, if valid, proceeds to run the evaluation process using
    the specified settings in the config file. It captures and logs any errors encountered during the
    evaluation process.

    Args:
        cfg (EvalModelConfig): Configuration object loaded from eval_model.yaml.

    Raises:
        BaseException: Captures and logs any exceptions that occur during the evaluation of the model.
    """
    validate_config(cfg)

    try:
        eval_model.run_trainer(cfg)
    except BaseException as err:
        logging.error(f"{type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
