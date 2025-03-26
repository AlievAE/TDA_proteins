from gudhi.representations import  PersistenceImage
from gudhi.representations import DiagramSelector
import numpy as np
import os
import pickle
from tqdm import tqdm
import json


def extract_from_pickles(folder_path, image_resolution=[20, 20]):
    extracted_data = []
    ind = []

    for i, file_name in enumerate(tqdm(os.listdir(folder_path))):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path) and file_name.endswith('.pkl'):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and len(data) == 1:
                    key = next(iter(data))
                    barcode = np.array(data[key]['barcodes'])
                    pds  = DiagramSelector(use=True).fit_transform(barcode)
                    vpdtr = np.vstack(pds)

                    pers = vpdtr[:,1]-vpdtr[:,0]
                    im_bnds = [np.min(vpdtr[:,0]), 0.3, np.min(pers), np.max(pers)]
                    PI_params = {'bandwidth': 0.1, 'weight': lambda x: x[1], 
                        'resolution': image_resolution, 'im_range': im_bnds}
                    PI = PersistenceImage(**PI_params).fit_transform(pds)
                    extracted_data.append([PI.tolist(), data[key]['label_MF'], data[key]['label_BP'], data[key]['label_CC']])

                else:
                    print(f"Skipping {file_name}: Expected a dictionary with one key, found {len(data)} keys.")
                ind.append(file_name)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    return extracted_data, ind

folder_path = "OneDrive/Desktop/barcodes_unprocessed/barcodes_valid_corrected"
results, ind = extract_from_pickles(folder_path)

with open("tda_val.json", "w") as json_file:
    json.dump(results, json_file)

with open("tda_val_ind.json", "w") as json_file:
    json.dump(ind, json_file)