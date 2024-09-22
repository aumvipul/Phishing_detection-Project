import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load phishing dataset
data = pd.read_csv('data.csv')

# Use relevant columns as features (these columns represent relevant features)
X = data[['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 
          'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain',
          'SSLfinal_State', 'Domain_registeration_length', 'Google_Index', 'DNSRecord', 'Page_Rank']]

# Target column (phishing or not)
y = data['Result']  # Assuming 'Result' column is the target, where 1 = phishing, 0 = legitimate

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'phishing_model.pkl')

# Function to extract features for inference
def extract_features(data_row):
    features = []
    features.append(data_row['having_IP_Address'])
    features.append(data_row['URL_Length'])
    features.append(data_row['Shortining_Service'])
    features.append(data_row['having_At_Symbol'])
    features.append(data_row['double_slash_redirecting'])
    features.append(data_row['Prefix_Suffix'])
    features.append(data_row['having_Sub_Domain'])
    features.append(data_row['SSLfinal_State'])
    features.append(data_row['Domain_registeration_length'])
    features.append(data_row['Google_Index'])
    features.append(data_row['DNSRecord'])
    features.append(data_row['Page_Rank'])
    return features
