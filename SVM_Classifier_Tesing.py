from data_preprocessing import read_csv
import numpy as np

from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

labels, features = read_csv('/Users/zhiyunjerrydeng/AEROPlan/embeddings_256_training.csv')

print(labels.shape)
print(features.shape)


## SVM Classifier

X_train = features
y_train = labels

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train the SVM classifier
clf = SVC(C=1.0, kernel='rbf', gamma='scale')
clf.fit(X_train_scaled, y_train)

## Tesing the classifier

y_test, X_test = read_csv('/Users/zhiyunjerrydeng/AEROPlan/embeddings_256_label.csv')
# y_test, X_test = read_csv('/Users/zhiyunjerrydeng/AEROPlan/embeddings_old.csv')
print(y_test)

# Scale the testing data using the same scaler as the training data
X_test_scaled = scaler.transform(X_test)

# Predict labels for the testing set
y_pred = clf.predict(X_test_scaled)

# Evaluate the classifier
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Detailed performance report
report = classification_report(y_test, y_pred)
print(report)

