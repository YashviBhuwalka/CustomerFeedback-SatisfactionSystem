import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('C:\\Users\\hp\\Downloads\\customer_feedback_satisfaction.csv')

# Display information about the data
print(data.info())
print(data.head())

# Visualizations
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

# Encode categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Country'] = le.fit_transform(data['Country'])
data['FeedbackScore'] = le.fit_transform(data['FeedbackScore'])
data['LoyaltyLevel'] = le.fit_transform(data['LoyaltyLevel'])

# Define features (X) and target (y)
X = data.drop(columns=['CustomerID', 'SatisfactionScore'])
y = data['SatisfactionScore']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
linear_model = LinearRegression()

# Fit the Linear Regression model
linear_model.fit(X_train, y_train)

# Predict with the trained model
y_pred_linear = linear_model.predict(X_test)

# Calculate metrics for the linear regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Display metrics for the linear regression model
print(f"Linear Regression Model - Mean Squared Error: {mse_linear:.2f}")
print(f"Linear Regression Model - Mean Absolute Error: {mae_linear:.2f}")
print(f"Linear Regression Model - R² Score: {r2_linear:.2f}")
