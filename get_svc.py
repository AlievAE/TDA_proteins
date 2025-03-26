from sklearn.decomposition import FastICA
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

#BP
print("Doing BP")
X_train, X_test, y_train, y_test = helper.split_dataset(X, y_BP, test_size=.3)
pipeline_linear_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('ica', FastICA(n_components=n_components, random_state=42, max_iter=3_000)),
    ('clf', MultiOutputClassifier(SVC(kernel='linear', random_state=42)))
])

pipeline_linear_svc.fit(X_train, y_train)
y_pred = pipeline_linear_svc.predict(X_test)
print(f"Metric for BP: {helper.count_f1_max(y_pred, y_test):.5f}")

#MF
print("Doing MF")
X_train, X_test, y_train, y_test = helper.split_dataset(X, y_MF, test_size=.3)
pipeline_linear_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('ica', FastICA(n_components=n_components, random_state=42, max_iter=3_000)),
    ('clf', MultiOutputClassifier(SVC(kernel='linear', random_state=42)))
])

pipeline_linear_svc.fit(X_train, y_train)
y_pred = pipeline_linear_svc.predict(X_test)
print(f"Metric for MF: {helper.count_f1_max(y_pred, y_test):.5f}")

#CC
print("Doing CC")
X_train, X_test, y_train, y_test = helper.split_dataset(X, y_CC, test_size=.3)
pipeline_linear_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('ica', FastICA(n_components=n_components, random_state=42, max_iter=3_000)),
    ('clf', MultiOutputClassifier(SVC(kernel='linear', random_state=42)))
])

pipeline_linear_svc.fit(X_train, y_train)
y_pred = pipeline_linear_svc.predict(X_test)
print(f"Metric for CC: {helper.count_f1_max(y_pred, y_test):.5f}")