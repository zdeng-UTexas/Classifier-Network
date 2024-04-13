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

# File path for saving the PGM file
target_image_path = 'global_costmap_P2.pgm'
# Writing the normalized data to a PGM file in the P2 format
with open(target_image_path, 'w') as f:
    # Write the PGM header
    f.write("P2\n") # Magic number for P2 format
    f.write(f"{normalized_cost_map.shape[1]} {normalized_cost_map.shape[0]}\n") # Width and height
    f.write("255\n") # Maximum pixel value
    # Write the pixel data
    for row in normalized_cost_map:
        f.write(' '.join(str(pixel) for pixel in row) + '\n')

def convert_p2_to_p5(p2_path, p5_path):
    # Read the P2 file
    with open(p2_path, 'r') as file:
        # Read and skip the magic number
        magic = file.readline()
        # Read the next lines for width, height, and max value
        dimensions = file.readline().strip()
        maxval = file.readline().strip()

        # Initialize a list to hold the pixel values
        pixels = []
        # Read the rest of the file for the pixel data
        for line in file:
            pixels.extend(line.strip().split())

    # Convert pixel values from strings to integers
    pixel_values = [int(pixel) for pixel in pixels]

    # Convert pixel values to bytes
    binary_data = bytes(pixel_values)

    # Write the P5 file
    with open(p5_path, 'wb') as file:
        file.write(bytearray(f"P5\n{dimensions}\n{maxval}\n", 'ascii'))
        file.write(binary_data)

# Specify the source P2 and target P5 file paths
source_p2_path = 'global_costmap_P2.pgm'
target_p5_path = 'global_costmap_P5.pgm'
# Convert the P2 file to P5
convert_p2_to_p5(source_p2_path, target_p5_path)

# Visualization of feature maps

def save_feature_map_as_pgm(feature_map, filename, max_val=255):
    """
    Saves a feature map as a PGM file.
    
    Parameters:
        feature_map (np.array): The numpy array of the feature map.
        filename (str): The output filename.
        max_val (int): The maximum pixel value (default 255 for PGM format).
    """
    # Normalize feature_map to 0-255
    normalized_feature_map = 255 * (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
    normalized_feature_map = normalized_feature_map.astype(np.uint8)
    
    # Write to PGM (P2 format)
    with open(filename, 'w') as f:
        f.write("P2\n")
        f.write(f"{feature_map.shape[1]} {feature_map.shape[0]}\n")
        f.write(f"{max_val}\n")
        for row in normalized_feature_map:
            f.write(' '.join(str(pixel) for pixel in row) + '\n')

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

    # Save the feature map as a PGM file
    pgm_filename = f'feature_map_{class_label}.pgm'
    save_feature_map_as_pgm(feature_map, pgm_filename)

