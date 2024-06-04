from sklearn.model_selection import train_test_split


def data_split(df, test_size=0.2, random_state=42):
    """
    Splits the dataframe into training and testing sets.

    Parameters:
    df (pd.DataFrame): The input dataframe containing features and target.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Testing features.
    y_train (pd.Series): Training target.
    y_test (pd.Series): Testing target.
    """
    y = df['cid']
    X = df.drop(['cid'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
