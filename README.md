Predicting Disk Failure

Data is available [here](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data#downloadingTheRawTestData). The experiments used 2021 Q1 through 2024 Q3 (the latest available at the time of writing)

Data preprocessing steps described in paper. The files are too large to upload to this repository

models
- lstm_5day.py: LSTM model with 5-day sliding window
- random_forest.py: Random Forest
- naive_bayes.py: Naive Bayes Classifier
