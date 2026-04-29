# File: house_price_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("house_prices.csv")

print("Dataset Preview:")
print(df.head())

# ---------------------------
# 2. Features & Target
# ---------------------------
X = df[['area_sqft', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

# ---------------------------
# 3. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Linear Regression Model
# ---------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("\n=== Linear Regression ===")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# ---------------------------
# 5. Decision Tree Model
# ---------------------------
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n=== Decision Tree ===")
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
print("R2 Score:", r2_score(y_test, y_pred_dt))

# ---------------------------
# 6. Predict New House Price
# ---------------------------
new_house = [[1600, 3, 2, 5]]  # Example: area, bedrooms, bathrooms, age

lr_price = lr_model.predict(new_house)
dt_price = dt_model.predict(new_house)

print("\n=== New House Prediction ===")
print("Linear Regression Price:", int(lr_price[0]))
print("Decision Tree Price:", int(dt_price[0]))
