"""Config file for CryoVIT experiments."""
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


samples = [
    "BACHD",  # keep
    "dN17_BACHD",  # keep
    "Q109",  # keep
    "Q18",  # keep
    "Q20",  # keep
    "Q53",  # keep
    "Q53_KD",  # keep
    "Q66",  # keep
    "Q66_GRFS1",  # keep
    "Q66_KD",  # keep
    "WT",  # keep
    "cancer",  # keep
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
class Model:
    lr: float
    weight_decay: float = 1e-3
    losses: Tuple[Dict] = (dict(_target_="cryovit.models.losses.DiceLoss"),)
    metrics: Tuple[Dict] = (
        dict(_target_="cryovit.models.metrics.DiceMetric", threshold=0.5),
    )


@dataclass
class CryoVIT(Model):
    lr: float = 1e-4
    _target_: str = "cryovit.models.CryoVIT"


@dataclass
class Trainer:
    accelerator: str = "gpu"
    devices: str = 1
    precision: str = "16-mixed"
    callbacks: Optional[List] = None
    enable_checkpointing: bool = False
    _target_: str = "pytorch_lightning.Trainer"


@dataclass
class TrainerFit(Trainer):
    max_epochs: int = 50
    logger: Optional[List] = None

    log_every_n_steps: int = 1
    num_sanity_val_steps: int = 0


@dataclass
class TrainerEval(Trainer):
    logger: bool = False
    enable_model_summary: bool = False


@dataclass
class Dataset:
    _partial_: bool = True


@dataclass
class SingleSample(Dataset):
    split_id: Optional[int] = None
    sample: Sample = MISSING
    _target_: str = "cryovit.data_modules.SingleSampleDataModule"


@dataclass
class MultiSample(Dataset):
    split_id: Optional[int] = None
    sample: Tuple[Sample] = MISSING
    test_samples: Tuple[Sample] = ()
    _target_: str = "cryovit.data_modules.MultiSampleDataModule"


@dataclass
class LOOSample(Dataset):
    split_id: Optional[int] = None
    sample: Sample = MISSING
    all_samples: Tuple[Sample] = tuple(s for s in Sample)
    _target_: str = "cryovit.data_modules.LOOSampleDataModule"


@dataclass
class FractionalLOO(Dataset):
    split_id: int = MISSING
    sample: Sample = MISSING
    all_samples: Tuple[Sample] = tuple(s for s in Sample)
    _target_: str = "cryovit.data_modules.FractionalSampleDataModule"


@dataclass
class ExpPaths:
    exp_dir: Path
    tomo_dir: Path
    split_file: Path
    cryovit_root: Optional[Path] = None


@dataclass
class DataLoader:
    num_workers: int = 8
    prefetch_factor: Optional[int] = 1
    persistent_workers: bool = False
    pin_memory: bool = True
    batch_size: Optional[int] = None
    _target_: str = "torch.utils.data.DataLoader"
    _partial_: bool = True


@dataclass
class TrainModelConfig:
    exp_name: str = MISSING
    label_key: str = MISSING
    aux_keys: Tuple[str] = ()

    model: Model = MISSING
    trainer: TrainerFit = MISSING
    dataset: Dataset = MISSING
    exp_paths: ExpPaths = MISSING
    dataloader: DataLoader = field(default=DataLoader())


# @dataclass
# class EvalModelConfig:
#     exp_name: str = MISSING
#     label_key: str = MISSING
#     split_id: Optional[int] = None
#     samples: Tuple[Sample] = MISSING

#     exp_paths: ExpPaths = MISSING
#     model: Model = MISSING
#     trainer: TrainerEval = MISSING
#     dataloader: DataLoader = field(default=DataLoader())


cs = ConfigStore.instance()
cs.store(name="dino_features_config", node=DinoFeaturesConfig)

cs.store(group="model", name="cryovit", node=CryoVIT)
cs.store(group="trainer", name="trainer_fit", node=TrainerFit)
cs.store(group="trainer", name="trainer_eval", node=TrainerEval)

cs.store(group="dataset", name="single", node=SingleSample)
cs.store(group="dataset", name="multi", node=MultiSample)
cs.store(group="dataset", name="loo", node=LOOSample)
cs.store(group="dataset", name="fractional", node=FractionalLOO)

cs.store(name="train_model_config", node=TrainModelConfig)
# cs.store(name="eval_model_config", node=EvalModelConfig)
