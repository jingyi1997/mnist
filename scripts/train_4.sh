export PYTHONPATH=$ROOT:$PYTHONPATH
now=$(date +"%Y%m%d_%H%M%S")
GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name $1 --partition=HA_vechicle -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python -u train.py --loss_type softmax --ckpt experiment/softmax_var0_retrai --epochs 100 --lr_steps 20 40 60 80 --bs 512 --base_lr 0.001 --data_list meta/train_all.txt --lr_mults 0.8 --data_dir images/training --val_data_dir images/testing --val_data_list meta/test_all.txt --classnum 10 --vis_freq 10 --print_freq 10 --sample_feat True --var_weight 0 



