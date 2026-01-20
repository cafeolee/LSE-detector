import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
P_DATA_PATH = os.path.join(SCRIPT_DIR, "data.pickle")

data_dict = pickle.load(open(P_DATA_PATH, "rb"))

# Unwrap the data and convert to array so it works properly
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Split data --> 80/20
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test
y_predict = model.predict(X_test)
score = accuracy_score(y_predict, y_test)

print(f"Accuracy of the model: {score}")

# Save the model
f = open("model.pickle", "wb")
pickle.dump({"model": model}, f)
f.close()

