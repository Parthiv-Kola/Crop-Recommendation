# 🌾 Crop Recommendation System

A machine learning project that recommends the most suitable crop to grow based on environmental and soil parameters using the **XGBoost** algorithm.  
The project is modularized for reusability and future integration into an MLOps pipeline.

---

## 📁 Project Structure

```bash
crop_recommendation/
│
├── data/
│ └── Crop_recommendation.csv # Dataset
│
├── src/
│ ├── data_loader.py # Loads and splits dataset
│ ├── preprocess.py # Scales features and encodes labels
│ ├── model_train.py # Trains and saves XGBoost model
│ ├── evaluate.py # Evaluates model performance
│ ├── predict.py # Loads model and predicts new data
│ └── main.py # Runs full pipeline with user input
│
├── artifacts/
│ ├── models/ # Saved trained model
│ ├── scaler/ # Saved StandardScaler
│ └── encoder/ # Saved LabelEncoder
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

1. **Data Loading**  
   Loads the dataset and splits it into training and testing sets.

2. **Preprocessing**  
   - Scales numerical features using `StandardScaler`  
   - Encodes crop labels using `LabelEncoder`

3. **Model Training**  
   - Trains an `XGBoostClassifier` on preprocessed data  
   - Saves trained model and preprocessing artifacts  

4. **Evaluation**  
   Achieved **99.32% accuracy** on the test set.  
   Data leakage checked — none detected.

5. **Prediction**  
   - Takes new soil and climate parameters  
   - Preprocesses inputs  
   - Predicts the most suitable crop

---

## 🧠 Example Run

```bash
python main.py
```

## Sample Input:

```bash
Nitrogen (N): 90
Phosphorus (P): 42
Potassium (K): 43
Temperature (°C): 26
Humidity (%): 81
pH: 6.8
Rainfall (mm): 209
```

## Output:

```bash
🌾 Recommended Crop: rice
```

## 📊 Model Details

| Metric    | Value                                        |
| --------- | -------------------------------------------- |
| Algorithm | XGBoost Classifier                           |
| Accuracy  | 99.32%                                       |
| Features  | N, P, K, Temperature, Humidity, pH, Rainfall |
| Classes   | 22 different crop types                      |

Plots (feature distributions, correlations, etc.) were created locally using matplotlib and seaborn.
They are interactive, so they don’t appear on GitHub by default.
But I saved them in **artifacts/plots**


## 🧩 Tech Stack

- Python 3.13
- NumPy, Pandas
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Joblib


## 🧾 Requirements

- I used anaconda so i don't know if this would work but I mentioned the libraries I installed in the requirements file

Install dependencies:
```bash
pip install -r requirements.txt
```