import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("insurance.csv")

# Remove duplicates
df = df.drop_duplicates().copy()


# -----------------------------
# Encode categorical variables
# -----------------------------
df["is_smoker"] = df["smoker"].map({
    "yes": 1,
    "no": 0
})

df["is_female"] = df["sex"].map({
    "female": 1,
    "male": 0
})


# Southeast region
df["region_southeast"] = (df["region"] == "southeast").astype(int)


# -----------------------------
# BMI category
# -----------------------------
df["bmi_category_obese"] = (
    df["bmi"] >= 30
).astype(int)


# -----------------------------
# Select features
# -----------------------------
features = [
    "age",
    "is_female",
    "bmi",
    "children",
    "is_smoker",
    "region_southeast",
    "bmi_category_obese"
]

X = df[features]
y = df["charges"]


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------
# Scale numerical features
# -----------------------------
scaler = StandardScaler()

numerical_columns = ["age", "bmi", "children"]

X_train = X_train.copy()
X_test = X_test.copy()

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)


# -----------------------------
# Train model
# -----------------------------
model = LinearRegression()

model.fit(X_train, y_train)


# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("Model Performance")
print("-----------------")
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")


# -----------------------------
# Save model and scaler
# -----------------------------
joblib.dump(model, "insurance_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")