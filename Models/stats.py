import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Loading dataset
data = pd.read_csv('C:\\Users\\hp\\Downloads\\customer_feedback_satisfaction.csv')

# Demographic analysis
print("Gender Distribution:\n", data['Gender'].value_counts(normalize=True))
print("Country Distribution:\n", data['Country'].value_counts(normalize=True))
print("Age Stats:\n", data['Age'].describe())

#  Income analysis
print("Income Stats:\n", data['Income'].describe())
sns.histplot(data['Income'], bins=20, kde=True)
plt.title("Income Distribution")
plt.show()

# Product and service quality
print("Product Quality Distribution:\n", data['ProductQuality'].value_counts(normalize=True))
print("Service Quality Distribution:\n", data['ServiceQuality'].value_counts(normalize=True))

# 4. Purchase frequency
print("Purchase Frequency Stats:\n", data['PurchaseFrequency'].describe())
sns.histplot(x=data['PurchaseFrequency'],bins=20,kde=True)
plt.title("Purchase Frequency Boxplot")
plt.show()

#  Target Variable
print("Customer Satisfaction Distribution:\n", data['SatisfactionScore'].value_counts(normalize=True))
sns.histplot(x=data['SatisfactionScore'],bins=20,kde=True)
plt.title("Customer Satisfaction Distribution")
plt.show()



sns.scatterplot(x=data['FeedbackScore'], y=data['SatisfactionScore'])
plt.title("Feedback Score vs. Satisfaction")
plt.show()


le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Country'] = le.fit_transform(data['Country'])
data['FeedbackScore'] = le.fit_transform(data['FeedbackScore'])
data['LoyaltyLevel'] = le.fit_transform(data['LoyaltyLevel'])


corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()



