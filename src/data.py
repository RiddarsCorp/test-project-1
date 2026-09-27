import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["mw", "logp", "tpsa"]


def load_data(path="data/measurements.csv"):
    """Load descriptors and target, ready for modelling."""
    df = pd.read_csv(path).drop_duplicates(subset="smiles")
    X = StandardScaler().fit_transform(df[FEATURES].values)
    y = df["logS"].values
    return X, y
