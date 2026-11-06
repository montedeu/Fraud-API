# Commit the trained model to the repository

Training runs locally from a reproducible script; its output (the LightGBM model file plus a metadata file holding the Model Version, Threshold and evaluation metrics) is committed to the repository and copied into the service image. One commit therefore pins both the code and the exact model it serves, and CI needs no Kaggle credentials and never trains.

This is acceptable here because the model is a few megabytes and is trained on public, synthetic data, so the artifact contains nothing secret.

## Considered Options

- **Upload the artifact to a GitHub Release and download it during the image build.** Keeps binaries out of git history, but splits one Model Version across two places and adds a download step that can fail.
- **Train in CI.** Needs the Kaggle token as a CI secret, makes every build minutes slower, and makes the served model depend on CI runner nondeterminism.

## Consequences

Retraining means committing a new model file; git history keeps the previous Model Versions. Only the current model file lives in the working tree.
