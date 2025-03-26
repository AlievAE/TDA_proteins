from sklearn.decomposition import FastICA
from py_boost import SketchBoost
import helper
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

image_dir = './data/validation'
label_dir = './data/validation_labels'

y_MF, y_BP, y_CC = helper.load_dataset(image_dir, label_dir, cut_per_set=None)
n_components = 256
X = np.load('./data/vectorization/validation.npz', allow_pickle=True)
X = X['arr_0']

#MF
print("Doing MF")
X_train, X_test, y_train, y_test = helper.split_dataset(X, y_MF, test_size=.3)
model_mf = SketchBoost(
            loss='multilabel', metric='f1', ntrees=20_000,
            lr=.01, es=1_000, lambda_l2=1, gd_steps=10,
            min_data_in_leaf=10, max_bin=256, max_depth=5,
            verbose=1_000
        )

model_mf.fit(X_train, y_train, eval_sets=[{'X': X_test, 'y': y_test}])

y_pred = model_mf.predict(np.array(X_test))
print(f"Metric for MF: {helper.count_f1_max(y_pred, y_test):.5f}")

#BP
print("Doing BP")
X_train, X_test, y_train, y_test = helper.split_dataset(X, y_BP, test_size=.3)
model_bp = SketchBoost(
            loss='multilabel', metric='f1', ntrees=20_000,
            lr=.01, es=1_000, lambda_l2=1, gd_steps=10,
            min_data_in_leaf=10, max_bin=256, max_depth=5,
            verbose=1_000
        )

model_bp.fit(X_train, y_train, eval_sets=[{'X': X_test, 'y': y_test}])

y_pred = model_bp.predict(np.array(X_test))
print(f"Metric for BP: {helper.count_f1_max(y_pred, y_test):.5f}")

#CC
print("Doing CC")
X_train, X_test, y_train, y_test = helper.split_dataset(X, y_CC, test_size=.3)
model_bp = SketchBoost(
            loss='multilabel', metric='f1', ntrees=20_000,
            lr=.01, es=1_000, lambda_l2=1, gd_steps=10,
            min_data_in_leaf=10, max_bin=256, max_depth=5,
            verbose=1_000
        )

model_bp.fit(X_train, y_train, eval_sets=[{'X': X_test, 'y': y_test}])

y_pred = model_bp.predict(np.array(X_test))
print(f"Metric for CC: {helper.count_f1_max(y_pred, y_test):.5f}")