from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import os
import numpy as np
from sklearn.impute import SimpleImputer


SMART_COLUMNS = ['smart_5_raw', 'smart_196_raw', 'smart_197_raw', 'smart_198_raw', 
                 'smart_187_raw', 'smart_184_raw', 'smart_183_raw', 'smart_12_raw', 
                 'smart_4_raw', 'smart_194_raw', 'failure']


def preprocess_csv(input_csv_path, bins=10):
    print(f"Processing file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    df = df[SMART_COLUMNS]

    df.replace('NaN', np.nan, inplace=True)

    for col in SMART_COLUMNS[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(0, inplace=True)

    # Discretize continuous features into equal-frequency bins
    for col in SMART_COLUMNS[:-1]:
        if df[col].nunique() > bins:
            try:
                df[col] = pd.qcut(df[col], bins, duplicates='drop', labels=False)
            except Exception as e:
                print(f"Error during binning for column {col}: {e}")
                print(f"Column data:\n{df[col].head()}")
                raise
        else:
            df[col] = df[col].astype(int)

    return df

# Train
def train_naive_bayes(X, y):
    print("Training Naive Bayes model...")
    nb_model = GaussianNB()
    nb_model.fit(X, y)
    print("Training complete.")
    return nb_model

# Evaluate
def evaluate_model(model, X, y, thresholds=[0.3, 0.5, 0.7]):
    print("Evaluating model at different thresholds...")
    results = []
    
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probabilities for positive class
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        y_pred = (y_pred_proba > threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        far = fp / (tn + fp) if (tn + fp) > 0 else 0
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"False Alarm Rate (FAR): {far:.4f}")
        print(f"False Discovery Rate (FDR): {fdr:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        results.append({
            "Threshold": threshold,
            "FAR": far,
            "FDR": fdr,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
    
    return pd.DataFrame(results)

def main():
    data_folder = "data"
    train_years = [2021, 2022, 2023]
    test_year = 2024
    
    # Load training data
    train_files = [os.path.join(data_folder, f"{year}_ST4000DM000.csv") for year in train_years]
    train_data = pd.concat([preprocess_csv(file) for file in train_files], ignore_index=True)
    
    # Load testing data
    test_file = os.path.join(data_folder, f"{test_year}_ST4000DM000.csv")
    test_data = preprocess_csv(test_file)
    
    # Separate features and labels for training
    X_train = train_data.drop(columns=['failure']).values
    y_train = train_data['failure'].values
    
    # Separate features and labels for testing
    X_test = test_data.drop(columns=['failure']).values
    y_test = test_data['failure'].values
    
    # Train and evaluate Naive Bayes model
    nb_model = train_naive_bayes(X_train, y_train)
    results = evaluate_model(nb_model, X_test, y_test, thresholds=[0.3, 0.5, 0.7])
    
    # Print results
    print("\nThreshold Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
