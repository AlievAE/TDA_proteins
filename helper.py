import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split

def count_f1_max(pred, target):
	"""
	    F1 score with the optimal threshold, Copied from TorchDrug.

	    This function first enumerates all possible thresholds for deciding positive and negative
	    samples, and then pick the threshold with the maximal F1 score.

	    Parameters:
	        pred (Tensor): predictions of shape :math:`(B, N)`
	        target (Tensor): binary targets of shape :math:`(B, N)`
    """
	pred = torch.Tensor(pred)
	target = torch.Tensor(target)
	if target.sum() == 0:
		return 0.0
	order = pred.argsort(descending=True, dim=1)
	target = target.gather(1, order)
	precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
	recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
	is_start = torch.zeros_like(target).bool()
	is_start[:, 0] = 1
	is_start = torch.scatter(is_start, 1, order, is_start)
	
	all_order = pred.flatten().argsort(descending=True)
	order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
	order = order.flatten()
	inv_order = torch.zeros_like(order)
	inv_order[order] = torch.arange(order.shape[0], device=order.device)
	is_start = is_start.flatten()[all_order]
	all_order = inv_order[all_order]
	precision = precision.flatten()
	recall = recall.flatten()
	all_precision = precision[all_order] - \
	                torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
	all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
	all_recall = recall[all_order] - \
	             torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
	all_recall = all_recall.cumsum(0) / pred.shape[0]
	all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
	all_f1 = torch.nan_to_num(all_f1, nan=0.0)
	return all_f1.max().item()

def load_dataset(image_dir, label_dir, cut_per_set):
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npz')], key=lambda x: int(x[:-4]))
    #X = []
    y_MF = []
    y_BP = []
    y_CC = []
    
    for file_name in files[:cut_per_set]:
        img_path = os.path.join(image_dir, file_name)
        with np.load(img_path, allow_pickle=True) as img_data:
            image = np.reshape(img_data['images'], (1, 660, 50, 50))
            #X.append(np.reshape(image, (1, -1)))

        label_path = os.path.join(label_dir, file_name)
        with np.load(label_path, allow_pickle=True) as label_data:
            y_MF.append(label_data['MF'].reshape(1, -1))
            y_BP.append(label_data['BP'].reshape(1, -1))
            y_CC.append(label_data['CC'].reshape(1, -1))

    #X = np.concatenate(X, axis=0)
    y_MF = np.concatenate(y_MF, axis=0)
    y_BP = np.concatenate(y_BP, axis=0)
    y_CC = np.concatenate(y_CC, axis=0)
    
    return y_MF, y_BP, y_CC

def filter_labels(y_train, y_test):
	constant_labels_zero = np.where(y_train.sum(axis=0) == 0)[0] #all zeros
	constant_labels_one = np.where(y_train.sum(axis=0) == y_train.shape[0])[0] # all ones
	constant_labels = np.concatenate([constant_labels_zero, constant_labels_one])

	if len(constant_labels) > 0:
		y_train_filtered = np.delete(y_train, constant_labels, axis=1)
		y_test_filtered = np.delete(y_test, constant_labels, axis=1)
	else:
		y_train_filtered = y_train.copy()
		y_test_filtered = y_test.copy()
	return y_train_filtered, y_test_filtered

def split_dataset(X, y, test_size = .2):
	X_train, X_test, y_train, y_test = train_test_split(
    	X, y, test_size=test_size, random_state=42
	)

	y_train, y_test = filter_labels(y_train, y_test)

	return X_train, X_test, y_train, y_test