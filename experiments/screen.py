import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X = np.load("data/features.npy")
y = np.load("data/labels.npy")

# Разделение выборки без фиксированного random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
print("roc-auc:", roc_auc_score(y_test, proba))
