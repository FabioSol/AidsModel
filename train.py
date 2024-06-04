from pipeline import create_pipeline
from evaluate_model import evaluate_model
import pickle


def train(X_train, X_test, y_train, y_test, hyperparams):
    model_pipeline = create_pipeline(**hyperparams)
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    metrics = evaluate_model(y_pred,y_test)


