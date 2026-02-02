# Evaluation
***

## Overview
Our dataset consists of 9 distinct classes which your model will aim to predict. Once your training is complete, the model will be evaluated by comparing your predictions against the ground truth labels.

## Scoring Metric: Macro F1-Score
***
We use the Macro-averaged F1-Score as our primary evaluation metric. Because the testing data is significantly unbalanced, this metric ensures that every class—regardless of size—is treated with equal importance in the final score.

## Technical Definitions
***
The evaluation process calculates the performance of each class individually using the following formulas:

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

The F1-Score for each class is then calculated as the harmonic mean of Precision and Recall: 

F1-Score = (2 * Precision * Recall) / (Precision + Recall)

## Final Calculation
***
The final leaderboard score is the average of the F1-Scores across all classes. This approach requires your model to perform well on every category, not just the most frequent ones.