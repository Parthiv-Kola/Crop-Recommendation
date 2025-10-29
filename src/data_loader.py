import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset and validate basic structure."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError("Missing required column: 'label'")
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split dataset into train and test sets."""
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_file = '/home/project_development/Projects/crop_recomendation/data/Crop_recommendation.csv'
    df = load_data(data_file)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f" Loaded {df.shape[0]} rows, split into {X_train.shape[0]} train and {X_test.shape[0]} test.")
