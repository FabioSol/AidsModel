from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


class Model(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.model = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

