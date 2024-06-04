from sklearn.pipeline import Pipeline
from AidsModel.preprocessing import Preprocess
from AidsModel.model import Model


def create_pipeline(**kwargs):
    pipeline = Pipeline([
        ('preprocessing', Preprocess()),
        ('model', Model(**kwargs))
    ])
    return pipeline
