import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('C:\\Users\\hp\\Downloads\\customer_feedback_satisfaction.csv')

# Data exploration
print(data.info())
print(data.head())

# Data visualization
plt.figure(figsize=(10, 6))
sns.histplot(data['SatisfactionScore'], bins=20, kde=True)
plt.title('Distribution of Satisfaction Score')
plt.xlabel('Satisfaction Score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='LoyaltyLevel', y='SatisfactionScore', data=data)
plt.title('Satisfaction Score by Loyalty Level')
plt.xlabel('Loyalty Level')
plt.ylabel('Satisfaction Score')
plt.show()

# Encoding categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Country'] = le.fit_transform(data['Country'])
data['FeedbackScore'] = le.fit_transform(data['FeedbackScore'])
data['LoyaltyLevel'] = le.fit_transform(data['LoyaltyLevel'])

# Define features and target variable
X = data.drop(columns=['CustomerID', 'SatisfactionScore'])
y = data['SatisfactionScore']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base learners
base_learners = [
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=2, max_features='sqrt', random_state=42)),
    ('svr', SVR(kernel='rbf', C=1, epsilon=0.1))
]

# Define the meta-model (blender)
meta_model = LinearRegression()

# Create the stacking model
stacked_model = StackingRegressor(estimators=base_learners, final_estimator=meta_model)

# Train the stacking model
stacked_model.fit(X_train, y_train)

# Make predictions with the stacked model
y_pred_stacked = stacked_model.predict(X_test)

# Evaluate the model
mse_stacked = mean_squared_error(y_test, y_pred_stacked)
mae_stacked = mean_absolute_error(y_test, y_pred_stacked)
r2_stacked = r2_score(y_test, y_pred_stacked)

print(f"Stacked Model - Mean Squared Error: {mse_stacked:.2f}")
print(f"Stacked Model - Mean Absolute Error: {mae_stacked:.2f}")
print(f"Stacked Model - R² Score: {r2_stacked:.2f}")
