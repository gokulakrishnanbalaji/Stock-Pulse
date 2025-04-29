import argparse
import os
import numpy as np
from sklearn.utils import resample

import logging

# Set up logging configuration
logging.basicConfig(
    filename='pipeline.log',              # Log file path
    level=logging.INFO,              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def balance_data(X, y, strategy="oversample"):
    """
    Balance a binary classification dataset using oversampling or undersampling.

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Binary target vector (0s and 1s).
    strategy : str, optional (default="oversample")
        Balancing strategy: "oversample" to increase minority class samples,
        or "undersample" to reduce majority class samples.

    Returns:
    --------
    X_balanced : numpy.ndarray
        Balanced feature matrix.
    y_balanced : numpy.ndarray
        Balanced target vector.
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be NumPy arrays")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary values (0 or 1)")
    if strategy not in ["oversample", "undersample"]:
        raise ValueError("strategy must be 'oversample' or 'undersample'")

    # Split data into positive and negative classes
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    
    # Apply balancing strategy
    if strategy == "oversample":
        if len(X_pos) > len(X_neg):
            # Oversample negative class to match positive class
            X_neg = resample(X_neg, replace=True, n_samples=len(X_pos), random_state=42)
        else:
            # Oversample positive class to match negative class
            X_pos = resample(X_pos, replace=True, n_samples=len(X_neg), random_state=42)
    elif strategy == "undersample":
        if len(X_pos) > len(X_neg):
            # Undersample positive class to match negative class
            X_pos = resample(X_pos, replace=False, n_samples=len(X_neg), random_state=42)
        else:
            # Undersample negative class to match positive class
            X_neg = resample(X_neg, replace=False, n_samples=len(X_pos), random_state=42)

    # Combine balanced data
    X_balanced = np.concatenate([X_pos, X_neg], axis=0)
    y_balanced = np.array([1] * len(X_pos) + [0] * len(X_neg))

    return X_balanced, y_balanced

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Balance a binary classification dataset.")
    parser.add_argument(
        "--processed_data_path",
        type=str,
        default="data/processed/",
        help="Path to the directory containing processed X.npy and y.npy files"
    )
    parser.add_argument(
        "--balanced_data_path",
        type=str,
        default="data/balanced/",
        help="Path to the directory to save balanced X.npy and y.npy files"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["oversample", "undersample"],
        default="oversample",
        help="Balancing strategy: 'oversample' or 'undersample'"
    )
    args = parser.parse_args()

    # Load data
    x_path = os.path.join(args.processed_data_path, "X.npy")
    y_path = os.path.join(args.processed_data_path, "y.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Input files not found at {args.processed_data_path}")
    
    X = np.load(x_path)
    y = np.load(y_path)

    # Balance the dataset
    X_balanced, y_balanced = balance_data(X, y, strategy=args.strategy)

    # Ensure output directory exists
    os.makedirs(args.balanced_data_path, exist_ok=True)

    # Save balanced data
    np.save(os.path.join(args.balanced_data_path, "X.npy"), X_balanced)
    np.save(os.path.join(args.balanced_data_path, "y.npy"), y_balanced)
    logging.info(f"Balanced data saved to {args.balanced_data_path}")

if __name__ == "__main__":
    main()