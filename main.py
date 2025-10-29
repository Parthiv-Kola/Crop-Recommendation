from src.predict import load_artifacts, predict

if __name__ == "__main__":
    model_path = "/home/project_development/Projects/crop_recomendation/artifacts/models/xgb_model.joblib"
    scaler_path = "/home/project_development/Projects/crop_recomendation/artifacts/scaler/scaler.joblib"
    encoder_path = "/home/project_development/Projects/crop_recomendation/artifacts/encoder/label_encoder.joblib"

    # Order: N, P, K, temperature, humidity, ph, rainfall
    user_input = [
        float(input("Nitrogen (N): ")),
        float(input("Phosphorus (P): ")),
        float(input("Potassium (K): ")),
        float(input("Temperature (°C): ")),
        float(input("Humidity (%): ")),
        float(input("pH: ")),
        float(input("Rainfall (mm): "))
    ]

    model, scaler, encoder = load_artifacts(model_path, scaler_path, encoder_path)
    prediction = predict(model, scaler, encoder, user_input)

    print(f"\n🌾 Recommended Crop: {prediction}")
