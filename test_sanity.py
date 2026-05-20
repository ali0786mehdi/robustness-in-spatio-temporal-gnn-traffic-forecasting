import os, sys
from src import config
from src.data_loader import prepare_dataset
from run_baselines import run_sanity

config.get_device()
data_prepared = prepare_dataset(config.DATASETS['METR-LA']['path'])
mean, std = data_prepared['mean'], data_prepared['std']
run_sanity(data_prepared, 'METR-LA', mean, std)
