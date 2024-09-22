from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained phishing detection model
model = joblib.load('phishing_model.pkl')

# Define feature extraction function
def extract_features(data_row):
    # Extracting relevant features from the data_row
    features = [
        data_row['having_IP_Address'],
        data_row['URL_Length'],
        data_row['Shortining_Service'],
        data_row['having_At_Symbol'],
        data_row['double_slash_redirecting'],
        data_row['Prefix_Suffix'],
        data_row['having_Sub_Domain'],
        data_row['SSLfinal_State'],
        data_row['Domain_registeration_length'],
        data_row['Google_Index'],
        data_row['DNSRecord'],
        data_row['Page_Rank']
    ]
    return features

# Route to serve the home page
@app.route('/') 
def home():
    return render_template('index.html')  # Ensure 'index.html' is in a 'templates' folder

# Route to accept URL and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)
        url_data = data['url_data']  # Expecting a dictionary of URL features

        # Extract features from the provided data
        features = extract_features(url_data)

        # Convert to DataFrame for model input
        df = pd.DataFrame([features], columns=[
            'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain',
            'SSLfinal_State', 'Domain_registeration_length', 'Google_Index', 
            'DNSRecord', 'Page_Rank'
        ])

        # Make prediction
        prediction = model.predict(df)[0]

        # Prepare the response
        result = "Phishing URL detected" if prediction == 1 else "Legitimate URL detected"

        return jsonify({'message': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message

if __name__ == '__main__':
    app.run(debug=True)
