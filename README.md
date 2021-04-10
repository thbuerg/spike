<div align="center">
 
# Time Series Forecasting for Energy Storage Deployment (TimESD)
</div>

## Description   
Deep Learning model to predict spikes in energy demand based on historical environmental and power consumption data. 
Hyperparameters are controlled via [hydra](https://hydra.cc/) and Logging is done with [neptune](https://www.neptune.ai).

Model currently implemented is a LSTM taking the last 7 days of environmental (Temperature, Windspeeds, Rain) and power consumption data in 15-minute resolution. 
The model predicts the expected daily maximum in electrical load in a 15 minute interval between 7AM and 8PM.

Energy data is retrieved from German Energy Providers, in this case the [Thueringer Energienetze](https://www.thueringer-energienetze.com).
Weather data is retrieved from the German Weather Forecasting Service [Deutscher Wetterdienst](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/).

Currently data is limited to 1y of data.

## How to run   
First, install dependencies (before you do that, install Miniconda.)
```bash
# clone timESD   
git clone https://github.com/thbuerg/timeESD

# install dependencies
cd timESD
conda env create -f environment.yml

# install timESD
pip install -e .

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
