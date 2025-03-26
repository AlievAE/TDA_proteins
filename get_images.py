from gudhi.representations import PersistenceImage
from gudhi.representations import DiagramSelector
import os
import numpy as np
import pickle

def get_per_im(barcodes):
    pds  = DiagramSelector(use=True).fit_transform(barcodes)
    vpdtr = np.vstack(pds)

    pers = vpdtr[:,1]-vpdtr[:,0]
    im_bnds = [np.min(vpdtr[:,0]), 0.05, 0.8, np.max(pers)]
    PI_params = {'bandwidth': 8e-3, 'weight': lambda x: x[1], 
                 'resolution': [50,50], 'im_range': im_bnds}
    image = PersistenceImage(**PI_params).fit_transform(pds)
    return image

def read_one_file(file_path, barcodes = True, labels = False):
    with open(file_path, 'rb') as file:
        seq_data = pickle.load(file)
    for sequence, data in seq_data.items():
        if barcodes:
            barcodes = data['barcodes']
        if labels:
            label_MF = data['label_MF']
            label_BP = data['label_BP']
            label_CC = data['label_CC']
    ans = {}
    if barcodes:
        try:
            barcodes = np.array(barcodes)
        except ValueError:
            for idx, el in enumerate(barcodes):
                cur = len(el)
                if idx > 0:
                    diff = cur - prev
                    if diff < 0:
                        for _ in range(-diff):
                            barcodes[idx].append([0, 0, 0])
                    elif diff > 0:
                        for _ in range(diff):
                            barcodes[idx].append([0, 0, 0])
                prev = len(el)
            barcodes = np.array(barcodes)
        PI = get_per_im(barcodes)
        ans['images'] = PI
    if labels:
        label_MF = np.array(label_MF)
        label_CC = np.array(label_CC)
        label_BP = np.array(label_BP)
        ans['MF'] = label_MF
        ans['CC'] = label_CC
        ans['BP'] = label_BP
    return ans

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def read_all_files(dir_path, output_dir, barcodes=True, labels=False, verbose=True):
    files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))], key=lambda x: int(x[:-4]))
    total_files = len(files)
    for i, filename in enumerate(files, start=1):
        file_path = os.path.join(dir_path, filename)
        print(filename)
        result = read_one_file(file_path, barcodes=barcodes, labels=labels)

        base_name = os.path.splitext(filename)[0]
        output_file_path = os.path.join(output_dir, f"{base_name}.npz")
        np.savez(output_file_path, **result)

        if verbose:
            if i % 10 == 0:
                print(f"Current file is {i} out of {total_files}")
            if i % 200 == 0:
                clear()

if __name__ == "__main__":
    read_all_files("data/test_set", "data/test_labels", barcodes=False, labels=True)
