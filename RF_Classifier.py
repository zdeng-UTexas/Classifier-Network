from data_preprocessing import read_csv

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X and y are your feature matrix and labels respectively
y_train, X_train = read_csv('/Users/zhiyunjerrydeng/AEROPlan/embeddings_256_training.csv')
print(y_train)

# Instantiate the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# If you have a separate test set:
y_test, X_test = read_csv('/Users/zhiyunjerrydeng/AEROPlan/embeddings_256_testing.csv')
y_test, X_test = read_csv('/Users/zhiyunjerrydeng/AEROPlan/embeddings_256_label.csv')
# print(y_test)
# X_test, y_test = ... # Your test data
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")



# Assuming y_test is your actual labels and y_pred is the predictions from the classifier
# Let's create a DataFrame to neatly display them side by side
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Display the DataFrame
print(results_df)
