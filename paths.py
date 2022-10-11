import os.path as op

current_dir = op.abspath('.')
data_dir = op.join(current_dir, "data")
training_set = op.join(data_dir, "train.csv")
test_set = op.join(data_dir, "test.csv")