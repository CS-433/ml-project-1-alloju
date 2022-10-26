import os


current_dir = os.path.abspath(".")
data_dir = os.path.join(current_dir, "data")
training_set = os.path.join(data_dir, "train.csv")
test_set = os.path.join(data_dir, "test.csv")
prediction_dir = os.path.join(current_dir, "predictions")
os.makedirs(prediction_dir, exist_ok=True)
