from typing import Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData

from scvi import settings
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.train import TrainRunner

from ._module import CPAModule
from ._task import CPATrainingPlan


class ManualDataSplitter(DataSplitter):
    """Manual train validation test splitter"""

    def __init__(
        self,
        adata: AnnData,
        val_idx,
        train_idx,
        test_idx,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata = adata
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.val_idx = val_idx
        self.train_idx = train_idx
        self.test_idx = test_idx

    def setup(self, stage: Optional[str] = None):
        gpus, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )


class CPA(UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata,
        batch_keys_to_dim: dict,
        split_key: str = None,
        **nn_kwargs,
    ):
        """CPA model

        Parameters
        ----------
        adata : Anndata
            Must be properly registered

        batch_keys_to_dim : str
        Map between covariates and the associated number of categories

        split_key : str, optional
            Key used to split the data between train test and validation.
            This must correspond to a observation key for the adata, composed of values
            'train', 'test', and 'ood'. By default None
        """
        super().__init__(adata)
        self.n_genes = self.summary_stats["n_vars"]
        n_treatments = adata.obsm["treatments"].shape[-1]
        self.n_treatments = n_treatments
        self.module = CPAModule(
            n_genes=self.n_genes,
            n_treatments=n_treatments,
            batch_keys_to_dim=batch_keys_to_dim,
            **nn_kwargs,
        )
        val_idx, train_idx, test_idx = None, None, None
        if split_key is not None:
            test_idx = np.where(adata.obs.loc[:, split_key] == "ood")[0]
            train_idx = np.where(adata.obs.loc[:, split_key] == "test")[0]
            train_idx = np.where(adata.obs.loc[:, split_key] == "train")[0]
        self.val_idx = val_idx
        self.train_idx = train_idx
        self.test_idx = test_idx

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        manual_splitting = (
            (self.val_idx is not None)
            and (self.train_idx is not None)
            and (self.test_idx is not None)
        )
        if manual_splitting:
            data_splitter = ManualDataSplitter(
                self.adata,
                val_idx=self.val_idx,
                train_idx=self.train_idx,
                test_idx=self.test_idx,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        else:
            data_splitter = DataSplitter(
                self.adata,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        training_plan = CPATrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Returns the basal latent variable

        Parameters
        ----------
        adata : Optional[AnnData], optional
            [description], by default None
        indices : Optional[Sequence[int]], optional
            Optional indices, by default None
        batch_size : Optional[int], optional
            Batch size to use, by default None
        """

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_reconstruction_error(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        """Estimates the reconstruction error (AE) or ELBO (VAE)"""
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        reco = []
        for tensors in scdl:
            _, _, _reco = self.module(tensors)
            reco.append(_reco.cpu())
        reco = torch.cat(reco, dim=0)
        return reco.mean()

    @torch.no_grad()
    def predict(
        self,
        # treatments: torch.Tensor = None,
        # cell_types: torch.Tensor = None,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        """Counterfactual-friendly gene expression prediction
        # TODO: See if another signature makes more sense for better usability

        To produce counterfactuals,

        Returns
        -------
        Tuple
            Gene expression means and standard variations
        """
        assert self.module.loss_ae == "gauss"

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        mus = []
        stds = []
        for tensors in scdl:
            _mus, _stds = self.module.get_expression(tensors)
            mus.append(_mus.cpu())
            stds.append(_stds.cpu())
        return torch.cat(mus, dim=0).numpy(), torch.cat(stds, dim=0).numpy()

    def get_embeddings(self, dose=1.0):
        """Computes drug embeddings

        Parameters
        ----------
        dose : float, optional
            Drug dose, by default 1.0

        """
        treatments = dose * torch.eye(self.n_treatments, device=self.module.device)
        embeds = self.module.treatments_embed(treatments).detach().cpu().numpy()
        return embeds
