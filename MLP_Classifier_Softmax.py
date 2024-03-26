from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from data_preprocessing_Unity import read_csv
import matplotlib.pyplot as plt

# Predefined terrain costs
costs = {
    'building': 254,
    'canopy': 160,
    'grass': 80,
    'dirt': 10,
    'smooth_trail': 0
}

# Load your training data
labels, features = read_csv('/home/zhiyundeng/AEROPlan/experiment/20240320/training/embedding_of_patch_32.csv')

# Scale features
scaler = StandardScaler().fit(features)
features_scaled = scaler.transform(features)

# Initialize and train MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=1)
mlp.fit(features_scaled, labels)

# Load testing data (make sure this matches how your read_csv function is structured)
_, X_new = read_csv('/home/zhiyundeng/AEROPlan/experiment/20240320/testing/embedding_of_patch_32.csv')

# Scale the new testing features based on the scaler fitted to the training data
X_new_scaled = scaler.transform(X_new)

# Predict probabilities for the testing data
probabilities = mlp.predict_proba(X_new_scaled)

# Calculate interpolated costs
# Convert the class labels in 'costs' to correspond to the order in mlp.classes_
costs_array = np.array([costs[class_label] for class_label in mlp.classes_])
interpolated_costs = np.dot(probabilities, costs_array)

# Save results to CSV in the current directory
predicted_classes = mlp.classes_[np.argmax(probabilities, axis=1)]
results = pd.DataFrame({
    'Predicted_Class': predicted_classes,
    'Interpolated_Cost': interpolated_costs
})

# Saving the file to the current directory
results.to_csv('predicted_classes_and_costs.csv', index=False)




## without interpolation
# Predict classes for the testing data
predicted_classes = mlp.predict(X_new_scaled)

# Map the predicted classes to their costs
predicted_costs = np.array([costs[class_label] for class_label in predicted_classes])

# Save the predicted classes and their costs to a CSV file
results_direct_cost = pd.DataFrame({
    'Predicted_Class': predicted_classes,
    'Direct_Assigned_Cost': predicted_costs
})

# Saving the file to the current directory
results_direct_cost.to_csv('predicted_classes_and_direct_costs.csv', index=False)



print('done')




# Assuming the CSV file 'predicted_classes_and_costs.csv' is saved in the current directory and has been generated as described
results = pd.read_csv('predicted_classes_and_costs.csv')

# Extracting interpolated costs from the results
interpolated_costs = results['Interpolated_Cost'].values

# Reshape the costs into the original aerial image shape (44 rows x 57 columns)
cost_map = interpolated_costs.reshape(44, 57)

# Visualizing the global costmap
plt.figure(figsize=(10, 7))
plt.imshow(cost_map, cmap='Greys', interpolation='nearest')
plt.colorbar(label='Interpolated Cost')
plt.title('Global Costmap of the Area')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.savefig('global_costmap.png')




# Assuming the CSV file 'predicted_classes_and_costs.csv' is saved in the current directory and has been generated as described
results = pd.read_csv('predicted_classes_and_direct_costs.csv')
# Extracting interpolated costs from the results
interpolated_costs = results['Direct_Assigned_Cost'].values
# Reshape the costs into the original aerial image shape (44 rows x 57 columns)
cost_map = interpolated_costs.reshape(44, 57)
# Visualizing the global costmap
plt.figure(figsize=(10, 7))
plt.imshow(cost_map, cmap='Greys', interpolation='none')
plt.colorbar(label='Interpolated Cost')
plt.title('Global Costmap of the Area')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.savefig('global_costmap_without_interpolation.png')