#! /bin/sh

echo Starting.

echo $(hostname)
echo $(which python)
echo $(python -c 'import torch; print(f"found {torch.cuda.device_count()} gpus.")')
echo $CUDA_VISIBLE_DEVICES

python train.py \
 +trainer.num_sanity_val_steps=1 \
 +trainer.gpus=3 \
 +trainer.max_epochs=100 \
 experiment.batch_size=128 \
 experiment.learning_rate=0.001

echo Done with submission script