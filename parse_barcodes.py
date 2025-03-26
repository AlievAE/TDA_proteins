from huggingface_hub import hf_hub_download
import pandas as pd
import tarfile

REPO_ID = "ultracheese/tda_for_proteins"
FILENAME = "barcodes_valid_corrected.tar.gz"

file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type='dataset')

output_dir = "barcodes_unprocessed"

with tarfile.open(file_path, "r:gz") as tar:
    tar.extractall(path=output_dir)