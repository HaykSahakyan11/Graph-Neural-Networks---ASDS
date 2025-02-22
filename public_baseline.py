import os
import csv
import numpy as np

from src.predict import predict_seal_model
from src.config import CONFIG

config = CONFIG()

###################
# random baseline #
###################


# Load test samples
test_file_path = config.test_data
with open(test_file_path, "r") as f:
    reader = csv.reader(f)
    test_set = list(reader)
test_set = [element[0].split(" ") for element in test_set]
test_set = [[int(element[0]), int(element[1])] for element in test_set]

predictions = predict_seal_model(file_path=test_file_path)
predictions = zip(np.array(range(len(test_set))), predictions)

data_dir = config.data_dir
test_predictions_csv = os.path.join(data_dir, "test_predictions.csv")
with open(test_predictions_csv, "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(i for i in ["ID", "Predicted"])
    for row in predictions:
        csv_out.writerow(row)
    pred.close()
print(f"Predictions saved to {test_predictions_csv}")
