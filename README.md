# Pollinator identification challenge
Understanding pollinator activity is vital for biodiversity research and global food security, yet manual observation from field recordings is slow and impractical at scale. In this project, we use a large set of labeled images extracted from videos of flowers to train a machine-learning model that classifies the type of pollinator in case of a visit. The data might contain images without insects, visual conditions vary strongly across recordings, and the same scene can look very different depending on the camera angle. A key objective is therefore not only accurate classification, but also strong generalization, testing whether a model trained on one viewpoint can successfully recognize pollinators from another. In this project, we focus on the lack of misclassifications. Correctly identifying the pollinator species is important because different insects play different roles in ecosystems. Misclassifying them can lead to incorrect conclusions about species-specific behavior, interactions, and their contribution to pollination.

## Challenge objectives

The task in this competition is to accurately classify pollinators from images. The input data originally consist of 528 MP4 videos recorded from multiple angles. From these videos, images were extracted and annotated based on the pollinator visiting a flower. The images were then cropped to focus on the flower region.

After feature extraction, a structured dataset was created, with each sample containing approximately 1500 features. The output of the model should be the correct pollinator visiting the flower, or 0 in the case where no pollinator is present. The goal of the task is to classify pollinators as accurately as possible.


## Team members

HORNERO MERINO Marina (group leader)\
CHOJNACKA Dominika\
LEONARDI Raphael\
CARKA Krisa \
EL OTMANI Youssef \
BUSCH Frederic \
Contact: marina.hornero-merino@universite-paris-saclay.fr


## How to execute the files

To be able to execute the jupyter notebooks inside this Github repo and compete in the AI Challenge, you will need to install
a few librarires by performing inside this folder : 

 ```bash
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Done!"

```

## Starting Kit
The starting kit is provided in the Codabench competition. It is provided as a guide for the participants. It contains:

- ingestion program: this is the program we use to read the data in codabench
- data: it contains the data for the competition
- sample code submission with the model that 




