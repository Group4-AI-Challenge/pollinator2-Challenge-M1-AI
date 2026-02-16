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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

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
        # Initialize scaler, PCA, and Random Forest
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=44, n_jobs=-1)
        self.is_fitted = False

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
        X = train_data["X_train"]
        y = train_data["y_train"]
        
        print(f"[*] - Original feature shape: {X.shape}")
        
        # Step 1: Standardize features
        print("[*] - Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 2: Apply PCA to reduce dimensionality
        print("[*] - Applying PCA to keep 95% variance...")
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"[*] - Reduced feature shape: {X_pca.shape}")
        print(f"[*] - Number of components: {self.pca.n_components_}")
        
        # Step 3: Train Random Forest on PCA-reduced features
        print("[*] - Training Random Forest (200 trees, max_depth=None)...")
        self.clf.fit(X_pca, y)
        self.is_fitted = True

    def predict(self, X_test):
        """
        This function predicts labels on test data.

        Parameters
        ----------
        test_data: list
            contains test data

        Returns
        -------
        y: 1D numpy array
            predicted labels
        """

        print("[*] - Predicting test set using trained Classifier")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        X = X_test['X_test']
        print(f"[*] - Test feature shape: {X.shape}")
        
        # Step 1: Standardize using the same scaler from training
        X_scaled = self.scaler.transform(X)
        
        # Step 2: Transform using the same PCA from training
        X_pca = self.pca.transform(X_scaled)
        print(f"[*] - Transformed test shape: {X_pca.shape}")
        
        # Step 3: Predict using trained Random Forest
        y = self.clf.predict(X_pca)
        return y

    
