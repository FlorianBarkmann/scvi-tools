import pathlib
from dataclasses import dataclass
from typing import Optional

import hydra.utils
import pandas as pd
import scanpy as sc
from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import OmegaConf

import scvi.model


@dataclass
class Slurm(SlurmQueueConf):
    mem_gb: int = 16
    timeout_min: int = 720
    partition: str = "gpu"
    gres: Optional[str] = "gpu:1"


@dataclass
class ModelConfig:
    _target_: str


@dataclass
class TrainingConfig:
    max_epochs: int = 400


@dataclass
class ScVIConfig(ModelConfig):
    _target_: str = "scvi.model.SCVI"
    n_latent: int = 10
    n_layers: int = 1
    n_hidden: int = 128
    dropout_rate: float = 0.1
    prior_distribution: str = "sdnormal"


@dataclass
class CanSigConfig(ScVIConfig):
    _target_: str = "scvi.model.CanSig"
    n_cnv_latent: int = 10
    n_cnv_layers: int = 1
    n_cnv_hidden: int = 128
    n_cnv_dropout_rate: float = 0.1


@dataclass
class Config:
    data_path: str
    model: ModelConfig
    trainer: TrainingConfig = TrainingConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="hydra/launcher", name="slurm", node=Slurm, provider="submitit_launcher")
cs.store(group="model", name="cansig", node=CanSigConfig())
cs.store(group="model", name="scvi", node=ScVIConfig())


def read_data(data_path: str):
    return sc.read(data_path)


def get_latent(model, trainer_config: TrainingConfig):
    model.train(max_epochs=trainer_config.max_epochs)
    return model.get_latent_representation()


def save_latent(latent, index, path):
    latent = pd.DataFrame(latent, index)
    latent.to_csv(path)


def setup_adata(adata, config):
    if config._target_ == "scvi.model.Cansig":
        scvi.model.CanSig.setup_anndata(adata, cnv_key="X_cnv", layer="counts")
    elif config._target_ == "scvi.model.SCVI":
        scvi.model.SCVI.setup_anndata(adata, batch_key="sample", layer="counts")
    else:
        raise ValueError(f"Unknown model config {config._target_}.")


def save_history(model):
    try:
        cwd = pathlib.Path.cwd()
        loss_dir = cwd.joinpath("losses")
        loss_dir.mkdir()
        for k, df in model.history.items():
            df.to_csv(loss_dir.joinpath(f"{k}.csv"))
    except Exception as e:
        print(e)


def dump_config(config: Config):
    with open("config.yaml", "w") as f:
        OmegaConf.save(config, f)


@hydra.main(config_name="config", config_path=None, version_base="1.1")
def main(config: Config):
    dump_config(config)
    adata = read_data(config.data_path)
    setup_adata(adata, config.model)
    model = hydra.utils.instantiate(config.model, adata=adata)
    latent = get_latent(model, config.trainer)
    save_latent(latent, adata.obs_names, "latent.csv")
    save_history(model)


if __name__ == '__main__':
    main()
