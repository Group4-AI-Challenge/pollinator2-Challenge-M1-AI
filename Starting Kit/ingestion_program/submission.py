class Model:
    def fit(self, X, y):
        # optional: train your model
        pass

    def predict(self, X):
        return X.sum(axis=1)