import joblib
import numpy as np

def load_artifacts(model_path, scaler_path, encoder_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    return model, scaler, encoder

def predict(model, scaler, encoder, input_data):
    """
    input_data: list or array of shape (n_features,)
    Returns: predicted label (decoded)
    """
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred_encoded = model.predict(input_scaled)
    pred_label = encoder.inverse_transform(pred_encoded)
    return pred_label[0]