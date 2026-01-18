# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
from datetime import datetime as dt
import numpy as np
import sklearn as sk


class Scoring:
    """
    This class is used to compute the scores for the competition.

    Atributes:
        * start_time (datetime): The start time of the scoring process.
        * end_time (datetime): The end time of the scoring process.
        * reference_data (dict): The reference data.
        * ingestion_result (dict): The ingestion result.
        * ingestion_duration (float): The ingestion duration.
        * scores_dict (dict): The scores dictionary.
    """

    def __init__(self, name=""):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.reference_data = None
        self.ingestion_result = None
        self.ingestion_duration = None
        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def load_reference_data(self, reference_dir):
        """
        Load the reference data.

        Args:
            reference_dir (str): The reference data directory name.
        """
        print("[*] Reading reference data")
        # TODO: Load reference data here
        reference_data_file = os.path.join(reference_dir, "y_test.json")
        self.reference_data = json.load(reference_data_file)

    def load_ingestion_result(self, predictions_dir):
        """
        Load the ingestion result.

        Args:
            predictions_dir (str): The predictions directory name.
        """
        print("[*] Reading ingestion result")

        # TODO: Load ingestion result here
        ingestion_result_file = os.path.join(predictions_dir, "result.json")
        self.ingestion_result = json.load(ingestion_result_file) # we assume ingestion result is stored as a json file

    def compute_scores(self):
        """
        Compute the scores for the competition. We use micro to account for 
        class imbalance (essentially the same as accuracy)

        """
        print("[*] Computing scores")
        score = sk.metrics.f1_score(self.reference_data, 
                                    self.ingestion_result, average='micro')
        # TODO: Compute scores here
        self.scores_dict = {'score': score}

    def write_scores(self, output_dir):

        print("[*] Writing scores")
        score_file = os.path.join(output_dir, "scores.json")
        with open(score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))
