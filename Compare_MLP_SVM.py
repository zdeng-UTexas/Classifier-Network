import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from data_preprocessing import read_csv

# Placeholder for your dataset loading function
# Replace it with your actual function to load data
def load_data():
    labels, features = read_csv('/home/zhiyundeng/AEROPlan/experiment/20240302/training/embedding_of_patch_64.csv')
    return labels, features  # Make sure to return the loaded data

labels, features = load_data()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define parameter range for SVM and perform grid search
param_grid_svm = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=2)
grid_svm.fit(X_train_scaled, y_train)

# View the best parameters
print("Best parameters for SVM: ", grid_svm.best_params_)


# Train SVM with the best parameters
svm_model = grid_svm.best_estimator_
svm_pred = svm_model.predict(X_test_scaled)

# Initialize and train MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=1)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)

# Calculate precision
svm_precision = precision_score(y_test, svm_pred, average='weighted')
mlp_precision = precision_score(y_test, mlp_pred, average='weighted')

print(f"SVM Precision: {svm_precision}")
print(f"MLP Precision: {mlp_precision}")


# Visualize the comparison
plt.bar(['SVM', 'MLP'], [svm_precision, mlp_precision], color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Precision Score')
plt.title('Comparison of Model Precision')
plt.ylim(0, 1)  # Assuming precision is a value between 0 and 1
plt.show()
