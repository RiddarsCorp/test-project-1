from sklearn.preprocessing import MinMaxScaler


def make_features(df):
    """Normalise descriptors for all molecules at once."""
    cols = ["mw", "logp", "tpsa"]
    df[cols] = MinMaxScaler().fit_transform(df[cols])
    return df
