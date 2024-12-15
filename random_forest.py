from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import os
import numpy as np

SMART_COLUMNS = ['smart_5_raw', 'smart_196_raw', 'smart_197_raw', 'smart_198_raw', 
                 'smart_187_raw', 'smart_184_raw', 'smart_183_raw', 'smart_12_raw', 
                 'smart_4_raw', 'smart_194_raw', 'failure']

def preprocess_csv(input_csv_path):
    print(f"Processing file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    df.fillna(0, inplace=True)  # Replace NaN values with 0
    df = df[SMART_COLUMNS]  # Keep only necessary columns
    return df

# Train
def train_random_forest(X, y):
    print("Training Random Forest model...")
    
    # Handle class imbalance by computing class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=20, 
        class_weight=class_weight_dict, 
        random_state=42
    )
    rf_model.fit(X, y)
    print("Training complete.")
    return rf_model

# Evaluate Random Forest model
def evaluate_random_forest(model, X, y):
    print("Evaluating Random Forest model...")
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    far = fp / (tn + fp) if (tn + fp) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"False Alarm Rate (FAR): {far:.4f}")
    print(f"False Discovery Rate (FDR): {fdr:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return far, fdr, precision, recall, f1

# Analyze Feature Importance
def analyze_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importance)
    return feature_importance

def main():
    data_folder = "data"
    train_years = [2021, 2022, 2023]  # Years for training
    test_year = 2024  # Year for testing
    
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
    
    # Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    evaluate_random_forest(rf_model, X_test, y_test)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(rf_model, train_data.drop(columns=['failure']).columns)

if __name__ == "__main__":
    main()