from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
from data_loader import load_data, split_data

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

if __name__ == "__main__":
    data_file = "/home/project_development/Projects/crop_recomendation/data/Crop_recommendation.csv"
    model_path = "/home/project_development/Projects/crop_recomendation/artifacts/models/xgb_model.joblib"
    scaler_path = "/home/project_development/Projects/crop_recomendation/artifacts/scaler/scaler.joblib"
    encoder_path = "/home/project_development/Projects/crop_recomendation/artifacts/encoder/label_encoder.joblib"

    # Load data and artifacts
    df = load_data(data_file)
    X_train, X_test, y_train, y_test = split_data(df)

    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    model = joblib.load(model_path)

    # Transform features and encode labels
    X_test_scaled = scaler.transform(X_test.values)
    y_test_enc = encoder.transform(y_test)

    acc = evaluate(model, X_test_scaled, y_test_enc)
    print(f"✅ Model accuracy on test set: {acc:.4f}")
