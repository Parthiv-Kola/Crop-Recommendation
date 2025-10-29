from xgboost import XGBClassifier
import joblib
from pathlib import Path

def train_model(X_train, y_train, save_path="artifacts/models/xgb_model.joblib"):
    """Train XGBoost classifier and save model."""
    model = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model



if __name__ == "__main__":
    from data_loader import load_data, split_data
    from preprocess import preprocess

    data_file = "/home/project_development/Projects/crop_recomendation/data/Crop_recommendation.csv"
    model_path = "/home/project_development/Projects/crop_recomendation/artifacts/models/xgb_model.joblib"
    scaler_path = "/home/project_development/Projects/crop_recomendation/artifacts/scaler/scaler.joblib"

    df = load_data(data_file)
    X_train, X_test, y_train, y_test = split_data(df)
    
    X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, scaler, encoder = preprocess(
        X_train.values, X_test.values, y_train, y_test,
        scaler_path=scaler_path,
        encoder_path="/home/project_development/Projects/crop_recomendation/artifacts/encoder/label_encoder.joblib"
    )


    model = train_model(X_train_scaled, y_train_enc, save_path=model_path)
    print(f"Model trained and saved at {model_path}")
