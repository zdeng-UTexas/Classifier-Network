import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from data_preprocessing_Unity import read_csv
import matplotlib.pyplot as plt
from PIL import Image

# Predefined terrain costs
costs = {
    # 'building': 254,
    # 'canopy': 160,
    # 'grass': 80,
    # 'dirt': 10,
    # 'smooth_trail': 0

    # EER
    # 'smooth_concrete': 0,
    # 'rocky_concrete': 10,
    # 'grass': 80,
    # 'gravel': 100,
    # 'tree': 160,
    # 'shrubbery': 120

    # AHG
    # 'concrete': 0,
    # 'building': 254,
    # 'grass': 80,
    # 'tree': 160,
    # 'dirt': 120

    # Pease Park
    # 'concrete': 0,
    # 'canopy': 120,
    # 'grass': 200,
    # 'dirt': 30

    # GQ - Google Earth
    'Building': 254,
    'Bush': 240,
    'Canopy': 160,
    'Dirt': 10,
    'Grass': 80,
    'Sand': 10,
    'Smooth_Trial': 0

    # GQ - Drone
    # 'building': 254,
    # 'bush': 240,
    # 'canopy': 160,
    # 'dirt': 10,
    # 'grass': 80,
    # 'sand': 10,
    # 'smooth_Trial': 0,
    # 'barrier': 254,
    # 'asphalt': 0



    #    labels = np.array(['Building' if 'Building' in label else 
    #                    'Bush' if 'Bush' in label else 
    #                    'Canopy' if 'Canopy' in label else 
    #                    'Dirt' if 'Dirt' in label else
    #                    'Grass' if 'Grass' in label else
    #                    'Sand' if 'Sand' in label else
    #                    'Smooth_Trial' if 'Smooth_Trial' in label else
    #                    label for label in labels])

}

# Load your training data
# labels, features = read_csv('/home/zhiyundeng/aeroplan/experiment/20240402_lejeune_emount_training/embedding_of_patch_part_5.csv')
# labels, features = read_csv('/home/zhiyundeng/aeroplan/experiment/20240605_EER_64_training/embedding_of_patch.csv')
# labels, features = read_csv('/home/zhiyundeng/aeroplan/experiment/20240612_EER_16_training/embedding_of_patch.csv')
# labels, features = read_csv('/home/zhiyundeng/aeroplan/experiment/20240719_AHG_2_training/embedding_of_patch.csv')
# labels, features = read_csv('/home/zhiyundeng/aeroplan/experiment/20240728_Pease_Park_30_training/embedding_of_patch.csv')
# labels, features = read_csv('/home/zhiyundeng/aeroplan/experiment/20240812_GQ_C_training/embedding_of_patch.csv')
labels, features = read_csv('/home/zhiyundeng/aeroplan/experiment/20240808_QC_A_10_training/embedding_of_patch.csv')



# Scale features
scaler = StandardScaler().fit(features)
features_scaled = scaler.transform(features)

# Initialize and train MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=1)
mlp.fit(features_scaled, labels)

# Load testing data
# _, X_new = read_csv('/home/zhiyundeng/aeroplan/experiment/20240402_lejeune_emount_global_cost_map_batch/combined_embedding_of_patch.csv')
# _, X_new = read_csv('/home/zhiyundeng/aeroplan/experiment/20240605_EER_64/embedding_of_patch.csv')
# _, X_new = read_csv('/home/zhiyundeng/aeroplan/experiment/20240614_EER_square_16/embedding_of_patch.csv')
# _, X_new = read_csv('/home/zhiyundeng/aeroplan/experiment/20240719_AHG_2/embedding_of_patch.csv')
# _, X_new = read_csv('/home/zhiyundeng/aeroplan/experiment/20240728_Pease_Park_30/embedding_of_patch.csv')
# _, X_new = read_csv('/home/zhiyundeng/aeroplan/experiment/20240812_GQ_C/combined_embedding_of_patch.csv')
_, X_new = read_csv('/home/zhiyundeng/aeroplan/experiment/20240808_QC_A_10/embedding_of_patch.csv')

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
# cost_map = interpolated_costs.reshape(500, 500)
# cost_map = interpolated_costs.reshape(46, 31) # EER_64
# cost_map = interpolated_costs.reshape(92, 62) # EER_32
# cost_map = interpolated_costs.reshape(95, 95) # EER_square_32
# cost_map = interpolated_costs.reshape(184, 124) # EER_16
# cost_map = interpolated_costs.reshape(190, 190) # EER_square_16
# cost_map = interpolated_costs.reshape(50, 50) # AHG_10
# cost_map = interpolated_costs.reshape(250, 250) # AHG_2
# cost_map = interpolated_costs.reshape(100, 100) # Pease Park (3000, 30)
cost_map = interpolated_costs.reshape(216, 216) # QC (2160, 10)
# cost_map = interpolated_costs.reshape(360, 360) # QC (drone, 3600, 10)

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

# Assuming cost_map_transposed is already defined
# Normalize the cost_map so that the minimum value corresponds to 0 and the maximum value to 255
normalized_cost_map = 255 * (cost_map_transposed - np.min(cost_map_transposed)) / (np.max(cost_map_transposed) - np.min(cost_map_transposed))
normalized_cost_map = normalized_cost_map.astype(np.uint8)

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(5, 3), dpi=500)
# Display the image with a grayscale colormap
cax = ax.imshow(normalized_cost_map, cmap='gray', interpolation='nearest', aspect='auto')
# Remove axis labels and ticks
ax.axis('off')

# Add a color bar with label
cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
cbar.set_label('Grayscale Value')

# Set the color bar ticks to include the min and max values
cbar.set_ticks([0, 255])
cbar.set_ticklabels([0, 255])

# Save the figure
plt.savefig('global_costmap_with_255_colorbar.png', bbox_inches='tight', pad_inches=0)
plt.close()


# Save the image as a PNG
# image.save('global_costmap_transposed_500x500.png')
image.save('global_costmap_EER_transposed_46x31.png')

# File path for saving the PGM file
target_image_path = 'global_costmap_EER_P2.pgm'
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
source_p2_path = 'global_costmap_EER_P2.pgm'
target_p5_path = 'global_costmap_EER_P5.pgm'
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
    # feature_map = class_probs.reshape(46, 31).T # EER_64
    # feature_map = class_probs.reshape(184, 124).T # EER_16
    # feature_map = class_probs.reshape(92, 62).T # EER_16
    # feature_map = class_probs.reshape(95, 95).T # EER_square_32
    # feature_map = class_probs.reshape(190, 190).T # EER_square_16
    # feature_map = class_probs.reshape(50, 50).T # AHG_10
    # feature_map = class_probs.reshape(250, 250).T # AHG_2
    # feature_map = class_probs.reshape(100, 100).T # Pease Park (3000, 30)
    feature_map = class_probs.reshape(216, 216).T # QC (2160, 10)
    # feature_map = class_probs.reshape(360, 360).T # QC (2160, 10)
    
    # Visualize and save the feature map
    plt.figure(figsize=(10, 7))
    plt.imshow(feature_map, cmap='Greys', interpolation='nearest')
    plt.colorbar(label='Probability')
    plt.title(f'Feature Map for {class_label}')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    # Save the figure
    plt.savefig(f'A_feature_layer_{class_label}.png')
    plt.close()  # Close the figure to free memory

    # Save the feature map as a PGM file
    pgm_filename = f'A_feature_layer_{class_label}.pgm'
    save_feature_map_as_pgm(feature_map, pgm_filename)

