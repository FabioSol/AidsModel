from sklearn.pipeline import Pipeline
from preprocessing import Preprocess
from model import Model


def create_pipeline(**kwargs):
    pipeline = Pipeline([
        ('preprocessing', Preprocess()),
        ('model', Model(**kwargs))
    ])
    return pipeline
