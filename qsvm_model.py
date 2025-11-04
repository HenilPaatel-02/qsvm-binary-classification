# qsvm_model.py
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def build_qsvm_model(X_train, X_test, y_train, y_test):

    # Define quantum feature map
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='full')

    # Use FidelityStatevectorKernel which computes kernel values via statevector
    # overlaps. This avoids relying on external Aer providers and works with
    # circuit statevector simulation.
    quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)

    # Compute kernel matrices
    kernel_train = quantum_kernel.evaluate(X_train, X_train)
    kernel_test = quantum_kernel.evaluate(X_test, X_train)

    # Classical SVM using precomputed quantum kernel
    qsvm = SVC(kernel='precomputed')
    qsvm.fit(kernel_train, y_train)
    y_pred = qsvm.predict(kernel_test)

    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred
