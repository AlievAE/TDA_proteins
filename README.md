# ml_proj
## Data sources
source for official code (private):
https://github.com/ipmaria/ml2025_per_proteins

source for data archives with barcodes:
https://huggingface.co/datasets/ultracheese/tda_for_proteins/tree/main

source for data archives with PersistenceImages:
https://huggingface.co/datasets/MKK03/barcodes_tda_protein/tree/main

## python-scripts
script to get PersistenceImages:
get_images.py

script to get vectorization via InceptionV3:
get_vectorization_inception.py

script to get metrics for SVC:
get_svc.py

script to get metrics for pyboost:
get_pyboost.py

## Jupyter Notebooks
playground for PersistenceImages:
persistence_images.ipynb

experiments regarding classic ML (mainly dim reduction techniques):
persistence_image_basic_ml.ipynb

experiments regarding pyboost (mainly parameters finetuning):
persistence_image_pyboost.ipynb

try to train AE:
ae_try.ipynb