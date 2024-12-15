import os
import argparse
import pandas as pd
import numpy as np
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# SMART parameters determined through analysis of Spearman correlation
SMART_COLUMNS = ['smart_5_raw', 'smart_196_raw', 'smart_197_raw', 'smart_198_raw', 
                 'smart_187_raw', 'smart_184_raw', 'smart_183_raw', 'smart_12_raw', 
                 'smart_4_raw', 'smart_194_raw', 'failure']

def preprocess_and_save_to_parquet(input_csv_path, output_parquet_path, limit=None):
    print(f"Processing file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    df.fillna(0, inplace=True)  # Replace NaN values with 0
    df = df[SMART_COLUMNS]
    
    # Limit the dataset size if the `limit` option is specified
    if limit is not None:
        print(f"Limiting dataset size to {limit} rows")
        df = df.sample(n=limit, random_state=42)

    df.to_parquet(output_parquet_path, index=False)
    print(f"Saved preprocessed file to: {output_parquet_path}")

# Downsample data to handle class imbalance
def downsample_data(df):
    df_majority = df[df['failure'] == 0]
    df_minority = df[df['failure'] == 1]
    
    df_majority_downsampled = resample(df_majority, 
                                       replace=False, 
                                       n_samples=len(df_minority) * 2, # Downsample factor here
                                       random_state=42)
    
    # Combine downsampled majority class with minority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    print(f"Downsampled dataset size: {len(df_downsampled)}")
    return df_downsampled

# LSTM
def create_lstm_model(input_dim, neurons=300):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(None, input_dim), return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model

# Prepare data in 5-day windows
def create_5_day_window_data(df, timesteps=5):
    features, labels = [], []
    feature_columns = [col for col in df.columns if col != 'failure']
    
    for i in range(len(df) - timesteps):
        features.append(df.iloc[i:i+timesteps][feature_columns].values)  
        labels.append(df.iloc[i+timesteps]['failure'])  
        
    return np.array(features), np.array(labels)

# Train
def train_lstm(df, timesteps=5):
    X, y = create_5_day_window_data(df, timesteps)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"Feature shape: {X_train.shape}, Label shape: {y_train.shape}")
    
    input_dim = X_train.shape[2]
    model = create_lstm_model(input_dim)
    
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1, shuffle=True)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    return model

def test_on_new_data(model, test_csv_path, timesteps=5, threshold=0.5):
    """
    Preprocess and evaluate the model on new test data (2024).
    """
    print(f"Processing and testing on {test_csv_path}...")
    df = pd.read_csv(test_csv_path)
    df.fillna(0, inplace=True)
    df = df[SMART_COLUMNS]
    df = downsample_data(df)
    
    X, y = create_5_day_window_data(df, timesteps)
    
    # Predict on the test set
    y_pred_proba = model.predict(X)  # Predicted probabilities
    y_pred = (y_pred_proba > threshold).astype(int)  # Binarize predictions
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Calculate FAR and FDR
    far = fp / (tn + fp) if (tn + fp) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"False Alarm Rate (FAR): {far:.4f}")
    print(f"False Discovery Rate (FDR): {fdr:.4f}")
    
    return far, fdr

def main():
    parser = argparse.ArgumentParser(description="Train LSTM for disk failure prediction")
    parser.add_argument("--limit", type=int, help="Limit the dataset size to a specified number of rows")
    args = parser.parse_args()
    
    input_folder = ""
    for year in range(2021, 2024):
        input_csv_path = os.path.join(input_folder, f"{year}_ST4000DM000.csv")
        output_parquet_path = f"data/parquet/{year}_ST4000DM000.parquet"
        preprocess_and_save_to_parquet(input_csv_path, output_parquet_path, limit=args.limit)

    parquet_files = [f"data/parquet/{year}_ST4000DM000.parquet" for year in range(2021, 2024)]
    for file_path in parquet_files:
        print(f"Loading and processing {file_path}...")
        df = pd.read_parquet(file_path)
        
        df_downsampled = downsample_data(df)
        
        print(f"Training LSTM on data from {file_path}...")
        lstm_model = train_lstm(df_downsampled)
    
    lstm_model.save("lstm_disk_failure_model.h5")
    print("Trained LSTM model saved as lstm_disk_failure_model.h5")

    test_csv_path = os.path.join(input_folder, "2024_ST4000DM000.csv")
    far, fdr = test_on_new_data(lstm_model, test_csv_path)
    print(f"Evaluation on 2024 Data - FAR: {far:.4f}, FDR: {fdr:.4f}")

# Entry Point
if __name__ == "__main__":
    main()
