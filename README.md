<div align="center">
 
# Time Series Forecasting for Energy Storage Deployment (TimeESD)
</div>

## Description   
Deep Learning model to predict spikes in energy demand based on historical environmental and power consumption data. 
Hyperparameters are controlled via [hydra](https://hydra.cc/) and Logging is done with [neptune](https://www.neptune.ai).

Model currently implemented is a LSTM taking the last 7 days of environmental (Temperature, Windspeeds, ??) and power consumption data in 15-minute resolution.

## How to run   
First, install dependencies (before you do that, install Miniconda.)
```bash
# clone timESD   
git clone https://github.com/thbuerg/timeESD

# install timESD
cd timESD
conda env create -f environment.yml
```   
 Next, navigate to train.py and execute it.   
 ```bash
# module folder
cd timESD/timESD/

# run preprocessing
python preprocessing.py +experiment.filepath=/path/to/your/data/

# run module 
python train.py \
  +experiment.filepath=/path/to/your/data/data_normed.csv \
  +trainer.default_root_dir=/path/to/your/results/ \
  +trainer.num_sanity_val_steps=1 +trainer.max_epochs=100 trainer.gpus=[0] experiment.batch_size=128 experiment.learning_rate=0.001
```

## Citation   
```
@article{Time Series Forecasting for Energy Storage Deployment,
  title={timESD},
  author={Thore Buergel},
  journal={github},
  year={2021}
}
```   
