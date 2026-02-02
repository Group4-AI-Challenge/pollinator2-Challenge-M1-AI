# Overview of the Challenge
***


## Introduction
Understanding pollinator activity is vital for biodiversity research and global food security, yet manual observation from field recordings is slow and impractical at scale. In this project, we use a large set of labeled images extracted from videos of flowers to train a machine-learning model that classifies the type of pollinator in case of a visit. Correctly identifying the pollinator species is important because different insects play different roles in ecosystems. The data might contain images without insects, visual conditions vary strongly across recordings, and the same scene can look very different depending on the camera angle. A key objective is therefore not only accurate classification, but also strong generalization, testing whether a model trained on one viewpoint can successfully recognize pollinators from another. Misclassifying them can lead to incorrect conclusions about species-specific behavior, interactions, and their contribution to pollination.

## Competition Tasks
***
The task in this competition is to accurately classify pollinators from images. The input data originally consist of 528 MP4 videos recorded from multiple angles. From these videos, images were extracted and annotated based on the pollinator visiting a flower. The images were then cropped to focus on the flower region.

After feature extraction, a structured dataset was created, with each sample containing approximately 1500 features. The output of the model should be the correct pollinator visiting the flower, or 0 in the case where no pollinator is present. The goal of the task is to classify pollinators as accurately as possible.

## Competition Phases
***
There are two phases in this competition.

Phase 1: In the first phase, you will be provided with a training set together with the corresponding labels. You will be able to upload your solutions to Codabench and evaluate how well your model performs. A leaderboard will be available during this phase to compare your results with those of other participants.

Phase 2: In the second phase, a new test dataset will be used to evaluate the ability of your code to generalize. This dataset will contain images captured from angles that are different from those in Phase 1. You need to submit only one model of your choice in that round, and you will see your final score.

## How to join this competition?
***
- Login or Create Account on [<ins>Codabench</ins>](https://www.codabench.org/)
- Go to "My Submissions" tab
- Accept terms and conditions
- Click the Register button

## Submissions
***
This competition allows only result submissions. Participants can submit a result submission as instructed in the `Starting Kit` tab.
-  Go to "Starting Kit" tab
- Download the `Starting Kit`
- Run the README.ipynb code from the `Starting Kit`
- Submit the created zip file in "My  Submissions" tab

## Timeline
***
Phase 1: 03.02.2026 - 20.02.2026

Phase 2: 20.02.2026 - 16.03.2026


## Credits
***
#### Challenge development: 

Dominika Chojnacka, Université Paris-Saclay (France)

Raphael Leonardi, Université Paris-Saclay (France)

Youssef El Otmani, Université Paris-Saclay (France)

Marina Hornero Merino, Université Paris-Saclay (France)

Krisa Carka, Université Paris-Saclay (France)

Frederic Busch, Université Paris-Saclay (France)

#### Mentoring: 

Khuong Thanh Gia Hieu, Université Paris-Saclay (France)

Ihsan Ullah, Université Paris-Saclay (France)

Lisheng Sun-Hosoya, Université Paris-Saclay (France)

Anne-Catherine Letournel, Université Paris-Saclay (France)

#### Data provider:

INRAE

#### Platform:
Codabench

## Contact
***
marina.hornero-merino@universite-paris-saclay.fr