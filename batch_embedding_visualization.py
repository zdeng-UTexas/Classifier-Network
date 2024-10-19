# python batch_embedding_visualization.py \
#     --input_dir /path/to/input_csvs/ \
#     --output_dir /path/to/output_images/ \
#     --scale_factor 10 \
#     --perplexity 30 \
#     --n_iter 1000 \
#     --random_state 42 \
#     --init pca \
#     --save_visualization

# Embedding Dimension: Ensure that your CSV files have embeddings with the expected dimension (128 in this case). If not, the script will issue a warning.
# Image Path Format: The script assumes that image filenames follow the pattern grid_x_y.png. Modify the extract_grid_coords function if your filenames differ.
# Scale Factor: Adjust the --scale_factor based on your preference for image size. A higher scale factor results in larger images.
# t-SNE Parameters: Feel free to experiment with perplexity, n_iter, and init parameters to achieve optimal visualization results based on your data.
# Performance: t-SNE can be computationally intensive, especially with a large number of embeddings. Monitor your system's resources and adjust parameters accordingly.
# Visualization Flag: The --save_visualization flag is optional. Use it only if you need the matplotlib visualization images.


import pandas as pd
import numpy as np
import os
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Batch Process Embedding CSVs and Generate Representation Images (RGB & Grayscale) using t-SNE")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing CSV files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save representation images.')
    parser.add_argument('--scale_factor', type=int, default=10, help='Scale factor for resizing images for better visibility.')
    parser.add_argument('--perplexity', type=float, default=30.0, help='Perplexity parameter for t-SNE.')
    parser.add_argument('--n_iter', type=int, default=1000, help='Number of iterations for t-SNE.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility in t-SNE.')
    parser.add_argument('--init', type=str, default='pca', choices=['random', 'pca'], help='Initialization method for t-SNE.')
    parser.add_argument('--save_visualization', action='store_true', help='If set, save matplotlib visualization of the RGB image.')
    args = parser.parse_args()
    return args

def extract_grid_coords(image_path):
    """
    Extract grid coordinates from the image path.
    Assumes the image filename is in the format 'grid_x_y.png'.
    """
    basename = os.path.basename(image_path)
    match = re.match(r'grid_(\d+)_(\d+)\.png', basename)
    if match:
        x, y = int(match.group(1)), int(match.group(2))
        return x, y
    else:
        raise ValueError(f"Invalid image path format: {image_path}")

def normalize_dimension(dim):
    """
    Normalize a dimension to the range [0, 255].
    """
    min_val = np.min(dim)
    max_val = np.max(dim)
    if max_val - min_val == 0:
        return np.zeros_like(dim)
    normalized = 255 * (dim - min_val) / (max_val - min_val)
    return normalized

def generate_images(csv_path, output_dir, scale_factor, tsne_params, save_visualization=False):
    """
    Process a single CSV file to generate RGB and Grayscale representation images.
    """
    try:
        # Load the CSV without headers
        df = pd.read_csv(csv_path, header=None)

        # Check if CSV has at least two columns (image_path and embeddings)
        if df.shape[1] < 2:
            raise ValueError(f"CSV file {csv_path} does not contain enough columns.")

        # Extract image paths
        image_paths = df.iloc[:, 0].tolist()

        # Extract embeddings
        embeddings = df.iloc[:, 1:].values  # Shape: (num_patches, embedding_dim)
        embedding_dim = embeddings.shape[1]
        if embedding_dim != 128:
            print(f"Warning: Expected embedding dimension 128, but got {embedding_dim} in {csv_path}.")

        # Extract grid coordinates for all patches
        grid_coords = []
        for path in image_paths:
            try:
                coord = extract_grid_coords(path)
                grid_coords.append(coord)
            except ValueError as e:
                print(f"Skipping patch due to error: {e}")
                continue

        if not grid_coords:
            print(f"No valid grid coordinates found in {csv_path}. Skipping file.")
            return

        # Separate x and y coordinates
        grid_x = [coord[0] for coord in grid_coords]
        grid_y = [coord[1] for coord in grid_coords]

        # Determine grid size
        max_x = max(grid_x)
        max_y = max(grid_y)
        grid_width = max_x + 1
        grid_height = max_y + 1
        print(f"Processing {os.path.basename(csv_path)}: Grid size {grid_width} columns x {grid_height} rows")

        # Initialize t-SNE with desired parameters
        tsne = TSNE(
            n_components=3,
            perplexity=tsne_params['perplexity'],
            n_iter=tsne_params['n_iter'],
            random_state=tsne_params['random_state'],
            init=tsne_params['init']
        )

        # Apply t-SNE to reduce dimensions
        print(f"Applying t-SNE to {os.path.basename(csv_path)}...")
        embeddings_tsne = tsne.fit_transform(embeddings)  # Shape: (num_patches, 3)
        print("t-SNE dimensionality reduction completed.")

        # Normalize t-SNE dimensions to [0, 255] for RGB
        R = normalize_dimension(embeddings_tsne[:, 0])
        G = normalize_dimension(embeddings_tsne[:, 1])
        B = normalize_dimension(embeddings_tsne[:, 2])

        # Create empty 2D arrays for each RGB channel
        image_grid_R = np.zeros((grid_height, grid_width), dtype=np.uint8)
        image_grid_G = np.zeros((grid_height, grid_width), dtype=np.uint8)
        image_grid_B = np.zeros((grid_height, grid_width), dtype=np.uint8)

        # Populate the grids with RGB values
        for idx, (x, y) in enumerate(zip(grid_x, grid_y)):
            image_grid_R[y, x] = R[idx]
            image_grid_G[y, x] = G[idx]
            image_grid_B[y, x] = B[idx]

        # Stack the RGB channels to create a color image
        image_grid_RGB = np.dstack((image_grid_R, image_grid_G, image_grid_B))

        # Convert the numpy array to a PIL Image
        color_image = Image.fromarray(image_grid_RGB, mode='RGB')

        # Resize the image for better visibility
        new_size = (color_image.width * scale_factor, color_image.height * scale_factor)
        color_image_resized = color_image.resize(new_size, Image.NEAREST)

        # Save the RGB image
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_rgb_path = os.path.join(output_dir, f"{csv_basename}_representation_rgb.png")
        color_image.save(output_rgb_path)
        print(f"RGB representation image saved to {output_rgb_path}")

        # Create Grayscale Representation
        print("Creating Grayscale representation image...")
        # Apply luminance formula: 0.2989 R + 0.5870 G + 0.1140 B
        grayscale_array = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)
        image_grid_gray = np.zeros((grid_height, grid_width), dtype=np.uint8)
        for idx, (x, y) in enumerate(zip(grid_x, grid_y)):
            image_grid_gray[y, x] = grayscale_array[idx]

        # Convert to PIL Image
        grayscale_image = Image.fromarray(image_grid_gray, mode='L')

        # Resize for better visibility
        grayscale_image_resized = grayscale_image.resize(new_size, Image.NEAREST)

        # Save the Grayscale image
        output_gray_path = os.path.join(output_dir, f"{csv_basename}_representation_gray.png")
        grayscale_image.save(output_gray_path)
        print(f"Grayscale representation image saved to {output_gray_path}")

        # Optional: Save matplotlib visualization
        if save_visualization:
            plt.figure(figsize=(8, 8))
            plt.imshow(color_image_resized)
            plt.title(f'RGB Representation Image: {csv_basename}')
            plt.axis('off')
            visualization_path = os.path.join(output_dir, f"{csv_basename}_visualization.png")
            plt.savefig(visualization_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Matplotlib visualization saved to {visualization_path}")

    def main():
        args = parse_arguments()

        input_dir = args.input_dir
        output_dir = args.output_dir
        scale_factor = args.scale_factor

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Collect all CSV files in the input directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {input_dir}. Exiting.")
            return

        print(f"Found {len(csv_files)} CSV files in {input_dir}. Starting batch processing...")

        # Define t-SNE parameters
        tsne_params = {
            'perplexity': args.perplexity,
            'n_iter': args.n_iter,
            'random_state': args.random_state,
            'init': args.init
        }

        # Process each CSV file with a progress bar
        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            csv_path = os.path.join(input_dir, csv_file)
            try:
                generate_images(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    scale_factor=scale_factor,
                    tsne_params=tsne_params,
                    save_visualization=args.save_visualization
                )
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

        print("Batch processing completed.")

    if __name__ == "__main__":
        main()
