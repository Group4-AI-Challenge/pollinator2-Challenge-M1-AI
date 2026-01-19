# Model file which contains a model class in scikit-learn style
# Model class must have these 3 methods
# - __init__: initializes the model
# - fit: trains the model
# - predict: uses the model to perform predictions
#
# Created by: Ihsan Ullah
# Created on: 13 Jan, 2026

# ----------------------------------------
# Imports
# ----------------------------------------
import sklearn as sk
from sklearn import linear_model

# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self):
        """
        This is a constructor for initializing classifier

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("[*] - Initializing Classifier")
        # TODO: complete this
        self.clf = linear_model.SGDClassifier(
    	loss="log_loss",
    	max_iter=10,
    	n_jobs=1
	)
        

    def fit(self, train_data):
        """
        This function trains the model provided training data

        Parameters
        ----------
        train_data: dict
            contains train data and labels

        Returns
        -------
        None
        """

        print("[*] - Training Classifier on the train set")
        # TODO: complete this
        X = train_data["X_train"]
        y = train_data["y_train"]
        self.clf.fit(X, y)

    def predict(self, test_data):

        """
        This function predicts labels on test data.

        Parameters
        ----------
        test_data: dict
            contains test data

        Returns
        -------
        y: 1D numpy array
            predicted labels
        """

        print("[*] - Predicting test set using trained Classifier")
        # TODO: complete this
        y = self.clf.predict(test_data["X_test"])
        return y
