# Competition Bundle

This is a sample competition bundle for M1 AI Challenge Class 2025-26. This bundle has the following pieces that make it work on Codabench.

- Competition.yaml
- Logo
- Pages
- Ingestion Program
- Scoring Program
- Input Data
- Reference Data

Each folder has its own `README` to explain what it does. Some folders only have README files and no other files. You are expected to fill them.

Ingestion and Scoring scripts have missing code that you have to fill to make sure everything works. Instructions are given for running the code locally to test if everything is ok before uploading the bundle to codabench.


### Compile Competition Bundle
You are given a utility function to make a zip of your bundle for Codabench. You can use this function as below.
```
python3 utilities/compile_bundle.py
```
⚠️ You can modify the zip function for your needs or zip the files directly. Make sure that you zip the bundle without the parent directory otherwise it will not work on Codabench.