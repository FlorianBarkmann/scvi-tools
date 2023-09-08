
adata = sc.read("neftel_prep.h5ad")
adata.obs["Cycling"] = adata.obs[["G1S", "G2M"]].max(1).map(
    lambda x: "cycling" if x > 0.0 else "G1")

# adata = adata[adata.obs["Cycling"] == "G1"].copy()
CanSig.setup_anndata(adata, cnv_key="X_cnv", layer="counts")
model = CanSig(adata)
model.train()

# %%

adata.obsm["latent"] = model.get_latent_representation()

sc.pp.neighbors(adata, use_rep="latent")
sc.tl.umap(adata)

# %%

adata.obs["MESlike"] = adata.obs[["MESlike1", "MESlike2"]].max(1)

adata.obs["NPClike"] = adata.obs[["NPClike1", "NPClike2"]].max(1)

adata.obs["Cycling"] = adata.obs[["G1S", "G2M"]].max(1)

sigs = pd.read_csv("~/Downloads/test.csv", index_col=0)
for col in sigs.columns:
    sc.tl.score_genes(adata, sigs[col].values, score_name=col)
# %%
adata.obs["celltype_score"] = adata.obs[["OPC", "AC", "MES", "NPC", "Cycling"]].idxmax(
    1)

# %%

sc.pl.umap(adata,
           color=["OPC", "AC", "MES", "NPC",
                  'sample', 'celltype_score', "G1S", "G2M"],
           ncols=2)

# %%


# %%

opc = pd.read_csv("../../annotations/glioblastoma/opc.csv", index_col=0)
npc1 = pd.read_csv("../../annotations/glioblastoma/npc1.csv", index_col=0)

# %%
len(set(opc.index).intersection(set(sigs["MES"].values)))

# %%

cnv_latent = model.get_cnv_latent_representation()

cdata = sc.AnnData(cnv_latent, obs=adata.obs)
cdata = cdata[~cdata.obs["subclones"].duplicated(), :].copy()
sc.pp.neighbors(cdata)
sc.tl.umap(cdata)

# %%

cdata.obs["n_subclone"] = cdata.obs["subclones"].str[-1]

sc.pl.umap(cdata, color=["sample", "n_subclone"])

# %%

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

adata = adata[adata.obs["celltype_score"] != "Cycling"].copy()
bio_conv = BioConservation(nmi_ari_cluster_labels_kmeans=False,
                           nmi_ari_cluster_labels_leiden=True)
batch_correction = BatchCorrection(pcr_comparison=False)
benchmark = Benchmarker(adata, label_key="celltype_score",
                        batch_key="sample",
                        embedding_obsm_keys=["latent"],
                        batch_correction_metrics=batch_correction,
                        bio_conservation_metrics=bio_conv)
benchmark.benchmark()

# %%
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

resolutions = [0.05 * i for i in range(2, 20)]
n_randoms = 5
aris = np.zeros((len(resolutions), n_randoms))
nmis = np.zeros((len(resolutions), n_randoms))

for random_seed in range(n_randoms):
    for n_res, res in enumerate(resolutions):
        sc.tl.leiden(adata, resolution=res, random_state=random_seed)
        aris[n_res, random_seed] = adjusted_rand_score(adata.obs["leiden"],
                                                       adata.obs["celltype_score"])
        nmis[n_res, random_seed] = normalized_mutual_info_score(adata.obs["leiden"],
                                                                adata.obs[
                                                                    "celltype_score"])
