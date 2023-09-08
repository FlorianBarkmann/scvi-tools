import scanpy as sc
import pandas as pd
import infercnvpy as cnv
from sklearn.metrics import silhouette_score

# %%

adata = sc.read("neftel_ss_cnv.h5ad")
adata = adata[adata.obs["cell_type"].notna()].copy()

# %%
highly_expressed_genes = adata.X.mean(0) > 0.1
bdata = adata[:, highly_expressed_genes].copy()
gene_order = pd.read_csv("../../annotations/gene_order.csv", index_col=0)
bdata.var = bdata.var.merge(gene_order, how="left", left_index=True, right_index=True)

cnv.tl.infercnv(bdata, reference_key="cell_type",
                reference_cat=['Macrophage', 'Oligodendrocyte', 'T_cell'],
                window_size=250)

# %%
adata.obsm["X_cnv"] = bdata.obsm["X_cnv"]
adata.uns["cnv"] = bdata.uns["cnv"]


# %%


adata = adata[adata.obs["cell_type"] == "Malignant"].copy()
adata.obs["subclones"] = None


for sample in adata.obs["sample"].unique():
    idx = adata.obs["sample"]==sample
    bdata = adata[idx].copy()
    cnv.tl.pca(bdata)
    cnv.pp.neighbors(bdata)
    best_score = -1.
    for res in [0.1 * i for i in range(1, 10)]:
        cnv.tl.leiden(bdata, resolution=res)
        if bdata.obs['cnv_leiden'].nunique() == 1:
            continue

        score = silhouette_score(bdata.obsm["X_cnv"], bdata.obs["cnv_leiden"])
        if score > best_score:
            print(f"new best socre {score} with {bdata.obs['cnv_leiden'].nunique()} clusters.")
            best_score = score
            adata.obs.loc[idx, "subclones"] = bdata.obs["sample"].astype(str) + bdata.obs["cnv_leiden"].astype(str)

# %%

cnvs = pd.DataFrame(adata.obsm["X_cnv"].todense(), index=adata.obs["subclones"])
cnvs = cnvs.groupby(level=0).mean()
adata.obsm["X_cnv"] = cnvs.loc[adata.obs["subclones"]].values

# %%

sc.pp.highly_variable_genes(adata, n_top_genes=4000, subset=True)

# %%


from scvi.model import CanSig


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

adata.obs["celltype_score"] = adata.obs[["MESlike", "NPClike", "OPClike", "AClike", "Cycling"]].idxmax(1)

# %%

sc.pl.umap(adata,
           color=["MESlike", "NPClike",  "AClike", "OPClike",
                  'sample', 'celltype_score', "G1S", "G2M"],
           ncols=2)

# %%


opc = pd.read_csv("../../annotations/glioblastoma/opc.csv", index_col=0)
npc1 = pd.read_csv("../../annotations/glioblastoma/npc1.csv", index_col=0)
