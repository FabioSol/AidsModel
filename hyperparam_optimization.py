from AidsModel.model import Model
import optuna
import pandas as pd
from AidsModel.evaluate_model import evaluate_model
from AidsModel.data_split import data_split


def optimize_hyperparameters(df: pd.DataFrame) -> optuna.study:
    study = optuna.create_study(direction='maximize')
    X_train, X_test, y_train, y_test = data_split(df)

    def objective(trial):
        # Suggest values for hyperparameters
        C = trial.suggest_loguniform('C', 1e-5, 1e2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        degree = trial.suggest_int('degree', 2, 5)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        model_ = Model(C=C, kernel=kernel, degree=degree, gamma=gamma)


        # Train the model
        model_.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model_.predict(X_test)
        metric = evaluate_model(y_test, y_pred)

        return metric

    study.optimize(objective, n_trials=100)

    return study.best_trial
