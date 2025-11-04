# main.py
from data_preprocessing import load_binary_iris
from qsvm_model import build_qsvm_model
from result_visualization import plot_results

def main():
    print("=== Quantum Support Vector Machine for Binary Classification ===")

    # Step 1: Load dataset
    X_train, X_test, y_train, y_test = load_binary_iris()
    print("Data loaded successfully.")

    # Step 2: Train QSVM model
    accuracy, y_pred = build_qsvm_model(X_train, X_test, y_train, y_test)
    print(f"QSVM Test Accuracy: {accuracy*100:.2f}%")

    # Step 3: Plot results
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()
