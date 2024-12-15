Predicting Disk Failure - ECE 225A Project

Data is available [here](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data#downloadingTheRawTestData). The experiments used 2021 Q1 through 2024 Q3 (the latest available at the time of writing)

Data preprocessing steps described in paper

data
- model_2021_ST4000DM000.csv
- model_2022_ST4000DM000.csv
- model_2023_ST4000DM000.csv
- model_2024_ST4000DM000.csv

models
- lstm_5day.py: LSTM model with 5-day sliding window
- random_forest.py: Random Forest
- naive_bayes.py: Naive Bayes Classifier
