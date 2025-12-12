"""
Task 1: Data pipeline (ETL)
- Reads data/data_sample.csv
- Cleans missing values
- Encodes categorical features
- Scales numeric features
- Writes train.csv and test.csv to data/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def generate_sample_csv(path):
    # small sample dataset if none exists
    df = pd.DataFrame({
        'age': [25, 30, np.nan, 45, 29, 34, 41, np.nan],
        'salary': [50000, 60000, 58000, 75000, 52000, np.nan, 90000, 62000],
        'department': ['sales', 'hr', 'hr', 'engineering', 'sales', 'engineering', np.nan, 'sales'],
        'target': [0,1,0,1,0,1,1,0]
    })
    df.to_csv(path, index=False)
    print(f"Sample CSV created at {path}")

def load_and_process(input_csv):
    df = pd.read_csv(input_csv)
    # Basic cleaning: fill numeric nan with median
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'target']

    # Define preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Fit-transform training, transform test
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Convert back to DataFrame (columns: numeric + ohe)
    ohe_cols = []
    if cat_cols:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        ohe_cols = list(cat_feature_names)
    final_cols = list(num_cols) + ohe_cols
    X_train_df = pd.DataFrame(X_train_p, columns=final_cols)
    X_train_df['target'] = y_train.reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test_p, columns=final_cols)
    X_test_df['target'] = y_test.reset_index(drop=True)

    return X_train_df, X_test_df

if __name__ == "__main__":
    sample_path = os.path.join(DATA_DIR, "sample_tabular.csv")
    if not os.path.exists(sample_path):
        from sklearn.impute import SimpleImputer
        generate_sample_csv(sample_path)
    # ensure SimpleImputer symbol is available in scope
    from sklearn.impute import SimpleImputer

    train_df, test_df = load_and_process(sample_path)
    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
    print("ETL complete. train.csv and test.csv written to data/")
