from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
import numpy as np


app = Flask(__name__)


model = TFDistilBertForSequenceClassification.from_pretrained('D:\CustomerAnalysis\Models\sentiment_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('D:\CustomerAnalysis\Models\sentiment_model')


@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    
   
    comment = data.get('comment')
    if not comment:
        return jsonify({'error': 'No comments provided'}), 400
    
   
    inputs = tokenizer(comment, truncation=True, padding=True, max_length=128, return_tensors='tf')
    
   
    logits = model(**inputs).logits
    predicted_class = np.argmax(logits.numpy(), axis=1)[0]  # Get the index of the max logit

    # Map the predicted class to the label (using your previous mapping)
    label_mapping = {0: "Positive", 1: "Negative", 2: "Neutral", 3: "Irrelevant"}
    predicted_label = label_mapping.get(predicted_class, "Unknown")
    
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
