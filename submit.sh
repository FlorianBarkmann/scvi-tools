
data_paths=$(find /cluster/work/boeva/fbarkmann/scvi-tools/data/ -name '*.h5ad' | paste -sd "," -)
python scvi_test.py hydra/launcher=slurm +data_path=$data_paths +model=scvi +model.prior_distribution=sdnormal,mixofgaus trainer.max_epochs=400,600,800 --multirun
python scvi_test.py hydra/launcher=slurm +data_path=$data_paths +model=cansig model.prior_distribution=sdnormal,mixofgaus model.n_cnv_layers=0,1,2 model.n_cnv_latent=5,10,15,20 --multirun
