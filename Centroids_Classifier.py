import numpy as np
from data_preprocessing import read_csv

y_train, X_train = read_csv('/Users/zhiyunjerrydeng/AEROPlan/embeddings_old.csv')
print(y_train)


# Compute centroids of each class in the training set
unique_classes = np.unique(y_train)
centroids = {cls: np.mean(X_train[y_train == cls], axis=0) for cls in unique_classes}

# Function to predict the class based on nearest centroid
def predict_class(X, centroids):
    predictions = []
    for sample in X:
        # Compute distances to each centroid
        distances = {cls: np.linalg.norm(sample - centroid) for cls, centroid in centroids.items()}
        # Find the class with the minimum distance
        predicted_class = min(distances, key=distances.get)
        predictions.append(predicted_class)
    return predictions

# Predict classes for the testing set

y_test, X_test = read_csv('/Users/zhiyunjerrydeng/AEROPlan/embeddings_old.csv')

y_pred = predict_class(X_test, centroids)

# print(y_pred)

# Now y_pred contains the predicted classes based on nearest centroid
