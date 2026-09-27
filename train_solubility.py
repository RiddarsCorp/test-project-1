import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Predict aqueous solubility (logS) from molecular descriptors.
df = pd.read_csv("data/descriptors.csv")
X = df.drop(columns=["logS"])
y = df["logS"]

# Standardize the descriptors before training.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

print("Test R2:", r2_score(y_test, model.predict(X_test)))
