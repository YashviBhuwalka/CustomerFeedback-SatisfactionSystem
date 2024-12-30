from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import logging

app = Flask(__name__)


model = joblib.load('D:\\CustomerAnalysis\\Models\\customer_satisfaction_analysis_model.pkl')
gender_encoder = joblib.load('D:\\CustomerAnalysis\\Models\\gender_encoder.pkl')
country_encoder = joblib.load('D:\\CustomerAnalysis\\Models\\country_encoder.pkl')
feedbackscore_encoder = joblib.load('D:\\CustomerAnalysis\\Models\\feedbackscore_encoder.pkl')
loyaltylevel_encoder = joblib.load('D:\\CustomerAnalysis\\Models\\loyaltylevel_encoder.pkl')


def safe_transform(encoder, value, default_label="unknown"):
    try:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            updated_classes = list(encoder.classes_) + [default_label]
            encoder.classes_ = np.array(updated_classes)
            return encoder.transform([default_label])[0]
    except Exception as e:
        logging.error(f"Error during encoding for value '{value}': {str(e)}")
        raise ValueError(f"Encoding error for value: {value}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        
        gender_encoded = safe_transform(gender_encoder, data['Gender'])
        country_encoded = safe_transform(country_encoder, data['Country'])
        feedbackscore_encoded = safe_transform(feedbackscore_encoder, data['FeedbackScore'])
        loyaltylevel_encoded = safe_transform(loyaltylevel_encoder, data['LoyaltyLevel'])

        
        features = [
            data['CustomerID'],  
            data['Age'],
            gender_encoded,
            data['Income'],
            data['ProductQuality'],
            data['ServiceQuality'],
            data['PurchaseFrequency'],
            feedbackscore_encoded,
            loyaltylevel_encoded
        ]

        
        prediction = model.predict([features])
        return jsonify({'Satisfaction level': prediction[0]})

    except KeyError as e:
        return jsonify({'message': f'Missing field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'message': f'Error processing the input: {str(e)}'}), 400
    except Exception as e:
        logging.error(f'Error: {str(e)}')
        return jsonify({'message': f'Error processing the request: {str(e)}'}), 400

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            
            data = pd.read_csv(file)

            
            required_columns = ['CustomerID', 'Age', 'Gender', 'Country', 'Income',
                                 'ProductQuality', 'ServiceQuality',
                                 'PurchaseFrequency', 'FeedbackScore', 'LoyaltyLevel']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return jsonify({'message': f'Missing required columns: {missing_columns}'}), 400

            
            data['Gender'] = data['Gender'].apply(lambda x: safe_transform(gender_encoder, x))
            data['Country'] = data['Country'].apply(lambda x: safe_transform(country_encoder, x))
            data['FeedbackScore'] = data['FeedbackScore'].apply(lambda x: safe_transform(feedbackscore_encoder, x))
            data['LoyaltyLevel'] = data['LoyaltyLevel'].apply(lambda x: safe_transform(loyaltylevel_encoder, x))

            
            features = data.drop(columns=['CustomerID'])
            predictions = model.predict(features)
            data['Predicted_Satisfaction'] = predictions

            return jsonify(data.to_dict(orient='records'))

        except Exception as e:
            logging.error(f'Error processing the file: {str(e)}')
            return jsonify({'message': f'Error processing the file: {str(e)}'}), 400
    else:
        return jsonify({'message': 'Invalid file format. Please upload a CSV file.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
