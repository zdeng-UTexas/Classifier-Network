from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
from data_preprocessing import read_csv
import numpy as np
import pandas as pd

# Load your data
labels, features = read_csv('/home/zhiyundeng/AEROPlan/experiment/20240302/training/embedding_of_patch_64.csv')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=1)
mlp.fit(X_train_scaled, y_train)

# Predict probabilities
probabilities = mlp.predict_proba(X_test_scaled)
predictions = mlp.predict(X_test_scaled)

# Assess uncertainty (using log loss here as an example)
uncertainty = log_loss(y_test, probabilities)

# Visualize uncertainty for a single prediction
# plt.hist(probabilities, bins=10, label=[f"Class {i}" for i in range(probabilities.shape[1])])
# plt.title("Prediction Uncertainty Visualization")
# plt.xlabel("Probability")
# plt.ylabel("Frequency")
# plt.legend()
# # plt.show()

# Calculate confidence scores for each class
confidence_scores = np.max(probabilities, axis=1) - (1.0 / probabilities.shape[1])

# Plot the confidence scores
plt.figure(figsize=(10, 6))
plt.hist(confidence_scores, bins=30, alpha=0.7, color='orange')
plt.title("Prediction Confidence Visualization")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.grid(True)
# plt.savefig("prediction_confidence_visualization.png")
# plt.show()


plt.savefig("prediction_uncertainty_visualization.png")

# MLP size details
input_dimension = X_train.shape[1]
output_dimension = len(np.unique(y_train))
number_of_layers = len(mlp.coefs_)  # Includes input, hidden, and output layers
hidden_layers_sizes = mlp.hidden_layer_sizes if isinstance(mlp.hidden_layer_sizes, tuple) else (mlp.hidden_layer_sizes,)

print(f"MLP Model Size Details:")
print(f"Input Dimension: {input_dimension}")
print(f"Output Dimension: {output_dimension}")
print(f"Number of Layers (including input and output): {number_of_layers}")
print(f"Sizes of Hidden Layers: {hidden_layers_sizes}")