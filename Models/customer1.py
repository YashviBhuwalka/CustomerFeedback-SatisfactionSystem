import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


data = pd.read_csv('C:\\Users\\hp\\Downloads\\customer_feedback_satisfaction.csv')


print(data.info())
print(data.head())


gender_encoder = LabelEncoder()
data['Gender'] = gender_encoder.fit_transform(data['Gender'])
joblib.dump(gender_encoder, 'D:\\CustomerAnalysis\\Models\\gender_encoder.pkl')

country_encoder = LabelEncoder()
data['Country'] = country_encoder.fit_transform(data['Country'])
joblib.dump(country_encoder, 'D:\\CustomerAnalysis\\Models\\country_encoder.pkl')

feedbackscore_encoder = LabelEncoder()
data['FeedbackScore'] = feedbackscore_encoder.fit_transform(data['FeedbackScore'])
joblib.dump(feedbackscore_encoder, 'D:\\CustomerAnalysis\\Models\\feedbackscore_encoder.pkl')

loyaltylevel_encoder = LabelEncoder()
data['LoyaltyLevel'] = loyaltylevel_encoder.fit_transform(data['LoyaltyLevel'])
joblib.dump(loyaltylevel_encoder, 'D:\\CustomerAnalysis\\Models\\loyaltylevel_encoder.pkl')


X = data.drop(columns=['CustomerID', 'SatisfactionScore'])
y = data['SatisfactionScore']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


best_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

base_learners = [
    ('rf', best_model),
    ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42))
]


meta_model = LinearRegression()


stacked_model = StackingRegressor(estimators=base_learners, final_estimator=meta_model)


stacked_model.fit(X_train, y_train)


y_pred_stacked = stacked_model.predict(X_test)
print(f"Stacked Model - MSE: {mean_squared_error(y_test, y_pred_stacked):.2f}")
print(f"Stacked Model - MAE: {mean_absolute_error(y_test, y_pred_stacked):.2f}")
print(f"Stacked Model - R²: {r2_score(y_test, y_pred_stacked):.2f}")


joblib.dump(stacked_model, 'D:\\CustomerAnalysis\\Models\\customer_satisfaction_analysis_model.pkl')
