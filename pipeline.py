from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model import Model

def create_pipeline(**kwargs):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', Model(**kwargs))
    ])
    return pipeline