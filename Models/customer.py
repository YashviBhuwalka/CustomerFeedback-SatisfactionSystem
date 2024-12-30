import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



data = pd.read_csv('C:\\Users\\hp\\Downloads\\customer_feedback_satisfaction.csv')


print(data.info())
print(data.head())





print(f"Dataset Shape: {data.shape}")


data.info()


data.describe()

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

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Country'] = le.fit_transform(data['Country'])
data['FeedbackScore'] = le.fit_transform(data['FeedbackScore'])
data['LoyaltyLevel'] = le.fit_transform(data['LoyaltyLevel'])


X = data.drop(columns=['CustomerID', 'SatisfactionScore'])
y = data['SatisfactionScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

best_model = RandomForestRegressor(
    n_estimators=300,          
    max_depth=20,              
    min_samples_split=10,      
    min_samples_leaf=2,        
    max_features='sqrt',       
    random_state=42           
)


best_model.fit(X_train, y_train)




y_pred = best_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean absolute Error: {mae:.2f}")

print(f"R² Score: {r2:.2f}")