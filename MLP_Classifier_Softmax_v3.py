import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from data_preprocessing_Unity import read_csv
import matplotlib.pyplot as plt
from PIL import Image

# Predefined terrain costs
costs = {
    'building': 254,
    'canopy': 160,
    'grass': 80,
    'dirt': 10,
    'smooth_trail': 0
}

# Load your training data
labels, features = read_csv('/home/zhiyundeng/aeroplan/experiment/20240402_lejeune_emount_training/embedding_of_patch_part_5.csv')

# Scale features
scaler = StandardScaler().fit(features)
features_scaled = scaler.transform(features)

# Initialize and train MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=1)
mlp.fit(features_scaled, labels)

# Load testing data
_, X_new = read_csv('/home/zhiyundeng/aeroplan/experiment/20240402_lejeune_emount_global_cost_map_batch/combined_embedding_of_patch.csv')
X_new_scaled = scaler.transform(X_new)

# Predict probabilities and calculate interpolated costs
probabilities = mlp.predict_proba(X_new_scaled)
costs_array = np.array([costs[class_label] for class_label in mlp.classes_])
interpolated_costs = np.dot(probabilities, costs_array)

# Saving the results
predicted_classes = mlp.classes_[np.argmax(probabilities, axis=1)]
results = pd.DataFrame({'Predicted_Class': predicted_classes, 'Interpolated_Cost': interpolated_costs})
results.to_csv('predicted_classes_and_costs.csv', index=False)

# Convert class labels to a format that can be included in a DataFrame for readability
class_labels = mlp.classes_
# Create a DataFrame with probabilities. Each column will correspond to a class label.
prob_df = pd.DataFrame(probabilities, columns=class_labels)
# Save the DataFrame to a CSV file
prob_df.to_csv('class_probabilities.csv', index=False)


# Load the saved results (if necessary, or directly use the arrays)
results = pd.read_csv('predicted_classes_and_costs.csv')
interpolated_costs = results['Interpolated_Cost'].values

# Correctly reshape the costs array. Ensure these dimensions match your actual data's structure.
cost_map = interpolated_costs.reshape(500, 500)

# If you need to swap the x and y axes (transpose the matrix)
cost_map_transposed = cost_map.T

# Visualization
plt.figure(figsize=(10, 7))
plt.imshow(cost_map_transposed, cmap='Greys', interpolation='nearest')
plt.colorbar(label='Interpolated Cost')
plt.title('Global Costmap of the Area')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.savefig('global_costmap_transposed.png')
plt.close()

# plt.figure(figsize=(1, 1), dpi=500)
# plt.imshow(cost_map_transposed, cmap='Greys', interpolation='nearest', aspect='auto')
# plt.axis('off')
# plt.savefig('global_costmap_transposed_500x500.png', bbox_inches='tight', pad_inches=0)
# plt.close()

# Normalize the cost_map to be within the 0-255 range for grayscale
normalized_cost_map = 255 * (cost_map_transposed - np.min(cost_map_transposed)) / (np.max(cost_map_transposed) - np.min(cost_map_transposed))
normalized_cost_map = normalized_cost_map.astype(np.uint8)
# Create an image from the normalized array
image = Image.fromarray(normalized_cost_map)
# Save the image as a PNG
image.save('global_costmap_transposed_500x500.png')


# Visualization of feature maps
# Load the probabilities from the CSV file saved previously
prob_df = pd.read_csv('class_probabilities.csv')

# Define the class ignore threshold
class_ignore_threshold = 0.025

# Iterate through each class and generate a feature map
for class_label in prob_df.columns:
    # Extract the probability array for the current class
    class_probs = prob_df[class_label].values
    
    # Apply the class ignore threshold
    class_probs[class_probs < class_ignore_threshold] = 0
    
    # Reshape the probabilities to match the dimensions of your area (e.g., 500x500)
    feature_map = class_probs.reshape(500, 500).T
    
    # Visualize and save the feature map
    plt.figure(figsize=(10, 7))
    plt.imshow(feature_map, cmap='Greys', interpolation='nearest')
    plt.colorbar(label='Probability')
    plt.title(f'Feature Map for {class_label}')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    # Save the figure
    plt.savefig(f'feature_map_{class_label}.png')
    plt.close()  # Close the figure to free memory