import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset from the CSV file
data = pd.read_csv('D:/Malware/Malware dataset.csv')

# Print unique values in the classification column
print("Unique values in the classification column:", data['classification'].unique())

# Convert the 'classification' column to numerical values
le = LabelEncoder()
data['classification'] = le.fit_transform(data['classification'])

# Split features and target variable
X = data.drop(columns=['classification'])  # Keep all columns except 'classification'
y = data['classification']  # Now 'classification' is numerical

# Check for non-numerical features and convert if necessary
print("Data types of the features:\n", X.dtypes)

# If there are non-numeric columns, convert them or drop them
X = X.select_dtypes(include=['int64', 'float64'])  # Keep only numerical features

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))



# Save the trained model
joblib.dump(model, 'malware_model.pkl')
print("Model training complete and saved as 'malware_model.pkl'.")
