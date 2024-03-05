import csv
import numpy as np

def read_csv(file_path):
    """
    Read a CSV file and return the labels and features as numpy arrays.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    labels_matrix (numpy.ndarray): Numpy array containing the labels.
    features_matrix (numpy.ndarray): Numpy array containing the features.
    """
    labels = []
    features = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            labels.append(row[0])
            features.append(list(map(float, row[1:])))

    labels = np.array(['dry_grass' if 'dry_grass' in label else 
                       'fresh_grass' if 'fresh_grass' in label else 
                       'shrubbery' if 'shrubbery' in label else 
                       'smooth_concrete' if 'smooth_concrete' in label else 
                       'tree' if 'tree' in label else 
                       label for label in labels])

    labels_matrix = np.array(labels)
    features_matrix = np.array(features)

    return labels_matrix, features_matrix

# Usage example
# file_path = '/Users/zhiyunjerrydeng/AEROPlan/embeddings_old.csv'
# labels, features = read_csv(file_path)

# print(labels.shape)
# print(features.shape)


