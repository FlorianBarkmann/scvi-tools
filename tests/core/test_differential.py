from functools import partial

import numpy as np
import pytest

from scvi.data import synthetic_iid
from scvi.model import SCVI
from scvi.model.base._utils import _prepare_obs
from scvi.utils import DifferentialComputation


def test_differential_computation(save_path):

    n_latent = 5
    adata = synthetic_iid()
    model = SCVI(adata, n_latent=n_latent)
    model.train(1)

    model_fn = partial(model.get_normalized_expression, return_numpy=True)
    dc = DifferentialComputation(model_fn, adata)

    cell_idx1 = np.asarray(adata.obs.labels == "label_1")
    cell_idx2 = ~cell_idx1

    dc.get_bayes_factors(cell_idx1, cell_idx2, mode="vanilla", use_permutation=True)
    dc.get_bayes_factors(cell_idx1, cell_idx2, mode="change", use_permutation=False)
    dc.get_bayes_factors(cell_idx1, cell_idx2, mode="change", cred_interval_lvls=[0.75])

    delta = 0.5

    def change_fn_test(x, y):
        return x - y

    def m1_domain_fn_test(samples):
        return np.abs(samples) >= delta

    dc.get_bayes_factors(
        cell_idx1,
        cell_idx2,
        mode="change",
        m1_domain_fn=m1_domain_fn_test,
        change_fn=change_fn_test,
    )

    # should fail if just one batch
    with pytest.raises(ValueError):
        model.differential_expression(adata[:20], groupby="batch")

    # test view
    model.differential_expression(
        adata[adata.obs["labels"] == "label_1"], groupby="batch"
    )

    # Test query features
    obs_col, group1, _, = _prepare_obs(
        idx1="(labels == 'label_1') & (batch == 'batch_1')", idx2=None, adata=adata
    )
    assert (obs_col == group1).sum() == adata.obs.loc[
        lambda x: (x.labels == "label_1") & (x.batch == "batch_1")
    ].shape[0]
    model.differential_expression(
        idx1="labels == 'label_1'",
    )
    model.differential_expression(
        idx1="labels == 'label_1'", idx2="(labels == 'label_2') & (batch == 'batch_1')"
    )

    # test that ints as group work
    a = synthetic_iid()
    a.obs["test"] = [0] * 200 + [1] * 200
    model = SCVI(a)
    model.differential_expression(groupby="test", group1=0)

    # test that string but not as categorical work
    a = synthetic_iid()
    a.obs["test"] = ["0"] * 200 + ["1"] * 200
    model = SCVI(a)
    model.differential_expression(groupby="test", group1="0")
