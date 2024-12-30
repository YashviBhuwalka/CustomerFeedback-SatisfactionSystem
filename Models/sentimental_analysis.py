

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, create_optimizer
from google.colab import files

# Upload the file
uploaded = files.upload()

# Check the uploaded file name (Google Colab's upload stores files in '/content/')
import os
file_name = list(uploaded.keys())[0]  # Get the uploaded file name
print(f"Uploaded file: {file_name}")

# Load dataset (Use the correct path to the uploaded file in Colab)
df = pd.read_csv(f'/content/{file_name}', header=None, names=['ID', 'Company', 'Label', 'Comment'])  # Use the correct file path
print(df.head())

# Fill null values in 'Comment' column with a default string
df['Comment'] = df['Comment'].fillna('No Comment')

# Map labels to numeric classes (positive=0, negative=1, neutral=2, irrelevant=3)
label_mapping = {"Positive": 0, "Negative": 1, "Neutral": 2, "Irrelevant": 3}
df['Label'] = df['Label'].map(label_mapping)

# Split data into input (X) and labels (y)
X = df['Comment'].tolist()  # Ensure this is a list of strings
y = df['Label'].tolist()

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the data using DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize training and test data (Make sure input is a list of strings)
train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'  # return PyTorch tensors (for extraction)
)

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'  # return PyTorch tensors (for extraction)
)

# Convert the PyTorch tensors to TensorFlow tensors
train_encodings_tf = {key: tf.convert_to_tensor(value.numpy()) for key, value in train_encodings.items()}
test_encodings_tf = {key: tf.convert_to_tensor(value.numpy()) for key, value in test_encodings.items()}

# Convert data into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings_tf, y_train)).batch(16)  # Adjust batch size if needed
test_dataset = tf.data.Dataset.from_tensor_slices((test_encodings_tf, y_test)).batch(16)

# Define the DistilBERT model for Sequence Classification
# Update the number of labels for multiclass classification (4 labels)
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

# Define optimizer, loss function, and metrics
batch_size = 16
epochs = 3
steps_per_epoch = len(train_dataset)
num_train_steps = steps_per_epoch * epochs
optimizer, schedule = create_optimizer(init_lr=2e-5, num_train_steps=num_train_steps, num_warmup_steps=0)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)

# Evaluate the model
results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Make predictions
y_pred_logits = model.predict(test_dataset).logits
y_pred = tf.argmax(y_pred_logits, axis=1).numpy()

# Evaluate predictions with confusion matrix and classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the fine-tuned model
model.save_pretrained('sentiment_model')
tokenizer.save_pretrained('sentiment_model')


