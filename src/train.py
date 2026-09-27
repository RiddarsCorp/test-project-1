import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/esol.csv")
X = data.drop(columns=["smiles", "logS"]).values
y = data["logS"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=300)
model.fit(X_train, y_train)
rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
print(f"RMSE on test: {rmse:.3f}")
# step 1
# step 2
# step 3
