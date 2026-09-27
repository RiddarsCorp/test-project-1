from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.data import load_data

SEED = 42


def run():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    print(f"R2 on held-out test: {score:.3f}")
    return score


if __name__ == "__main__":
    run()
