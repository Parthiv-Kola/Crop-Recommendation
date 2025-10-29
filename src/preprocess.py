from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

def preprocess(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler_path: str = None,
    encoder_path: str = None
):
    """Scale numerical features and encode labels."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Save artifacts
    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    if encoder_path:
        Path(encoder_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, encoder_path)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, encoder


if __name__ == "__main__":
    from data_loader import load_data, split_data

    data_file = '/home/project_development/Projects/crop_recomendation/data/Crop_recommendation.csv'
    scaler_path = '/home/project_development/Projects/crop_recomendation/artifacts/scaler/scaler.joblib'
    encoder_path = '/home/project_development/Projects/crop_recomendation/artifacts/encoder/label_encoder.joblib'

    df = load_data(data_file)
    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, scaler, encoder = preprocess(
        X_train.values, X_test.values, y_train, y_test,
        scaler_path=scaler_path, encoder_path=encoder_path
    )

    print(f"Data scaled and encoded.\nScaler saved at {scaler_path}\nEncoder saved at {encoder_path}")
