# Conda

The following steps will help you setup a conda environment.

1. Install Conda
https://conda.io/projects/conda/en/latest/user-guide/install/index.html

2. Create env
```
conda create --name m1-ai-challenge python=3.12.8
```

3. Activate env
```
conda activate m1-ai-challenge
```

4. Install required packages
```
pip install -r requirements.txt
```

5. (Optional) Deactivate and remove env
To deactivate or/and remove the env after you have used it, you can use the following commands
```
conda deactivate
```
```
conda env remove --name m1-ai-challenge
```
