# Pollinator Identification - 2026 Codabench AI Challenge

This project implements a Codabench AI challenge for accurately classifying images of pollinator species extracted from field recordings. The challenge aims at supporting biodiversity research as well as food security efforts, and is developed as part of the course "Creation of an AI challenge" at Université Paris Saclay in 2026. 

## Table of contents

- [Background](#background)
- [Challenge Objectives](#challenge-objectives)
- [Install](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Authors](#authors)
- [License](#license)

## Background

Understanding pollinator activity is vital for biodiversity research and global food security, yet manual observation from field recordings is slow and impractical at scale. In this project, we use a large set of labeled images extracted from videos of flowers to train a machine-learning model that classifies the type of pollinator in case of a visit. The data might contain images without insects, visual conditions vary strongly across recordings, and the same scene can look very different depending on the camera angle. A key objective is therefore not only accurate classification, but also strong generalization, testing whether a model trained on one viewpoint can successfully recognize pollinators from another. In this project, we focus on the lack of misclassifications. Correctly identifying the pollinator species is important because different insects play different roles in ecosystems. Misclassifying them can lead to incorrect conclusions about species-specific behavior, interactions, and their contribution to pollination.

## Challenge Objectives

The task in this competition is to accurately classify pollinators from images. The input data originally consist of 528 MP4 videos recorded from multiple angles. From these videos, images were extracted and annotated based on the pollinator visiting a flower. The images were then cropped to focus on the flower region.

After feature extraction, a structured dataset was created, with each sample containing approximately 1500 features. The output of the model should be the correct pollinator visiting the flower, or 0 in the case where no pollinator is present. The goal of the task is to classify pollinators as accurately as possible.

## Install

### 1. Clone the project

To get started, open a terminal to clone the repository and move into the project folder:

```bash
git clone https://github.com/Group4-AI-Challenge/pollinator2-Challenge-M1-AI.git
cd pollinator2-challenge-m1-ai
```

### 2. Install dependencies

It is recommended that you set up a virtual environment to isolate the projects' dependencies from other projects you are working on. To achieve this, you can install conda by following the instructions in [Option A: Conda Setup](#option-a-conda-setup). If you do not wish to do so, you can proceed with [Option B: Global Installation](#option-b-global-installation).

#### Option A: Conda Setup

1. Install conda from https://conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Create a new environment:
```
conda create --name pollinator-challenge python=3.12.8
```
3. Activate your environment:
```
conda activate pollinator-challenge
```
4. Install project dependencies: 
```
pip install -r "Starting Kit/requirements.txt"
```
5. (Optional) Deactivate and remove your environment:
```
conda deactivate
conda env remove --name pollinator-challenge
```

#### Option B: Global installation

To globally intall all dependencies, move to the "Starting Kit" folder and install all dependencies from the "requirements.txt" file by executing the following commands:

```bash
cd "Starting Kit"
echo "Installing dependencies..."
pip install -r requirements.txt

```

## Usage

### Starting Kit

Once installed, you are ready to familiarize yourself with the Starting Kit. The "Starting Kit" folder contains: 

- the "README.ipynb" file to get you started
- the training data in "input_data" 
- a sample model in "sample_code_submission"
- a sample model output in "sample_result_submission"
- the ingestion program in "ingestion_program" that automatically runs your model once submitted to Codabench

You can simply start by reading the "README.ipynb" file to begin working on your own submission for the competition.

## Project Structure

```bash
pollinator2-challenge-m1-ai/
├── README.md                        # Main project README
├── competition_bundle/              # Codabench competition package
│   ├── competition.yaml             # Competition configuration
│   ├── ingestion_program/           # Data ingestion pipeline
│   ├── scoring_program/             # Evaluation pipeline
│   ├── input_data/                  # Training data
│   ├── reference_data/              # Test data and labels
│   ├── sample_code_submission/      # Submission template
│   ├── sample_result_submission/    # Example submission
│   ├── utilities/                   # Build & packaging utilities
│   └── pages/                       # Codabench info pages
├── starting-kit/                    # Beginner-friendly workspace
│   ├── README.ipynb                 # Interactive starter notebook
│   ├── README.md                    # Quick start guide
│   ├── data/                        # Local development data
│   ├── ingestion_program/           # Local ingestion script
│   ├── sample_code_submission/      # Model template
│   ├── sample_result_submission/    # Result template
│   └── scoring_program/             # Local scoring script
├── phases/                          # Competition phases
│   ├── phase_1/                     # Phase 1 test data
│   └── phase_2/                     # Phase 2 test data
├── conda/                           # Environment setup
│   ├── README.md                    # Conda setup instructions
│   └── requirements.txt             # Python dependencies
```

## Authors

### Team Lead
- Marina Hornero Merino: marina.hornero-merino@universite-paris-saclay.fr

### Team Members
- Dominika Chojnacka
- Youssef El Otmani
- Krisa Carka
- Raphael Leonardi
- Frederic Busch

### Mentors
- Khuong Thanh Gia Hieu
- Ihsan Ullah
- Lisheng Sun

## License

This project is part of the M1 AI Challenge course at Université Paris-Saclay. All code and materials are provided for educational and competition purposes.