from model import Model
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def optimize_hyperparameters(X:pd.DataFrame,y:pd.DataFrame)->optuna.study:
    study = optuna.create_study(direction='maximize')

    def objective(trial):
        # Suggest values for hyperparameters
        C = trial.suggest_loguniform('C', 1e-5, 1e2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        degree = trial.suggest_int('degree', 2, 5)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        model_ = Model(C=C, kernel=kernel, degree=degree, gamma=gamma)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model_.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy
    study.optimize(objective, n_trials=100)
    return study
