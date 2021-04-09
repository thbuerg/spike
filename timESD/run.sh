#! /bin/sh

echo Starting.

echo $(hostname)
echo $(which python)
echo $(python -c 'import torch; print(f"found {torch.cuda.device_count()} gpus.")')
echo $CUDA_VISIBLE_DEVICES

python train.py \
 +trainer.default_root_dir=/home/buergelt/projects/timESD/results/
 +experiment.filepath=/home/buergelt/projects/timESD/data/data_normed.csv
 +trainer.num_sanity_val_steps=1 \
 trainer.gpus=-1 \
 experiment.batch_size=128 \
 experiment.learning_rate=0.001

echo Done with submission script