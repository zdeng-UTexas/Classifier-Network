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

    # labels = np.array(['smooth_trail' if 'smooth_trail' in label else 
    #                    'dirt' if 'dirt' in label else 
    #                    'grass' if 'grass' in label else 
    #                    'canopy' if 'canopy' in label else 
    #                    'building' if 'building' in label else 
    #                    label for label in labels])
    
    # EER
    # labels = np.array(['smooth_concrete' if 'smooth_concrete' in label else 
    #                    'rocky_concrete' if 'rocky_concrete' in label else 
    #                    'grass' if 'grass' in label else 
    #                    'tree' if 'tree' in label else 
    #                    'shrubbery' if 'shrubbery' in label else
    #                    'gravel' if 'gravel' in label else
    #                    label for label in labels])

    # AHG
    # labels = np.array(['concrete' if 'concrete' in label else 
    #                    'canopy' if 'canopy' in label else 
    #                    'grass' if 'grass' in label else 
    #                    'dirt' if 'dirt' in label else
    #                    label for label in labels])

    # GQ-Google Earth
    labels = np.array(['Building' if 'Building' in label else
                        'Bush' if 'Bush' in label else
                        'Canopy' if 'Canopy' in label else 
                        'Dirt' if 'Dirt' in label else
                        'Grass' if 'Grass' in label else
                        'Sand' if 'Sand' in label else
                        'Smooth_Trial' if 'Smooth_Trial' in label else
                        label for label in labels])
    # GQ-Drone
    # labels = np.array(['building' if 'building' in label else 
    #                    'canopy' if 'canopy' in label else 
    #                    'dirt' if 'dirt' in label else
    #                    'asphalt' if 'asphalt' in label else
    #                    'barrier' if 'barrier' in label else
    #                    label for label in labels])

    
    labels_matrix = np.array(labels)
    features_matrix = np.array(features)

    return labels_matrix, features_matrix

# Usage example
# file_path = '/Users/zhiyunjerrydeng/AEROPlan/embeddings_old.csv'
# labels, features = read_csv(file_path)

# print(labels.shape)
# print(features.shape)


