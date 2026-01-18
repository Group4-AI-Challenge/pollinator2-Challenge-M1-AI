# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
from datetime import datetime as dt
import numpy as np

def load_npy(file_dir, file_name):
    return np.load(os.path.join(file_dir, file_name))

class Ingestion:
    """
    Class for handling the ingestion process.

    Args:
        None

    Attributes:
        * start_time (datetime): The start time of the ingestion process.
        * end_time (datetime): The end time of the ingestion process.
        * model (object): The model object.
        * train_data (dict): The train data dict.
        * test_data (dict): The test data dict.
        * ingestion_result (dict): The ingestion result dict.
    """

    def __init__(self):
        """
        Initialize the Ingestion class.

        """
        self.start_time = None
        self.end_time = None
        self.model = None
        self.train_data = None
        self.test_data = None
        self.ingestion_result = None

    def start_timer(self):
        """
        Start the timer for the ingestion process.
        """
        self.start_time = dt.now()

    def stop_timer(self):
        """
        Stop the timer for the ingestion process.
        """
        self.end_time = dt.now()

    def get_duration(self):
        """
        Get the duration of the ingestion process.

        Returns:
            timedelta: The duration of the ingestion process.
        """
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stopped. Returning None")
            return None

        return self.end_time - self.start_time

    def save_duration(self, output_dir=None):
        """
        Save the duration of the ingestion process to a file.

        Args:
            output_dir (str): The output directory to save the duration file.
        """
        duration = self.get_duration()
        duration_in_mins = int(duration.total_seconds() / 60)
        duration_file = os.path.join(output_dir, "ingestion_duration.json")
        if duration is not None:
            with open(duration_file, "w") as f:
                f.write(json.dumps({"ingestion_duration": duration_in_mins}, indent=4))

    def load_train_and_test_data(self, input_dir):
        """
        Load the training and testing data.

        """
        print("[*] Loading Train data")


        # TODO: Load train and/or test data here
        tr_data_file = "X_train.npy"
        tr_labels_file = "y_train.npy"
        test_data_file = "X_test.npy"

        self.train_data ={ 
            'X_train': load_npy(input_dir, tr_data_file), 
            'y_train': load_npy(input_dir, tr_labels_file)
        }
        # self.train_data = load_npy(input_dir, tr_labels_file)
        self.test_data = {'X_test': load_npy(input_dir, test_data_file)}

    def init_submission(self, Model):
        """
        Initialize the submitted model.

        Args:
            Model (object): The model class.
        """
        print("[*] Initializing Submmited Model")
        # TODO: initialize your model here
        self.model = Model()

    def fit_submission(self):
        """
        Fit the submitted model.
        """
        print("[*] Fitting Submmited Model")
        # TODO: call the fit method of your submitted model
        self.model.fit(self.train_data)
        # self.model.fit()

    def predict_submission(self):
        """
        Make predictions using the submitted model.
        """
        print("[*] Calling predict method of submitted model")

        # TODO: Save the output from the model predict method. You can use this later in compute_result function
        self.y_test = self.model.predict(self.test_data['X_test'])

    def compute_result(self):
        """
        Compute the ingestion result.
        """
        print("[*] Computing Ingestion Result")

        # TODO: complete this function to compute set the output of the ingestion program
        # You can save the result in a dictionary to be saved as json file by save_result function

        # TODO: Modify below to set the result dict
        self.ingestion_result = {
            "predictions": self.y_test
        }

    def save_result(self, output_dir=None):
        """
        Save the ingestion result to files.

        Args:
            output_dir (str): The output directory to save the result files.
        """
        result_file = os.path.join(output_dir, "result.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(result_file, "w") as f:
            f.write(json.dumps(self.ingestion_result, indent=4))
