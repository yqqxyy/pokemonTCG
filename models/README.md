# Model manifests

Model binaries are intentionally excluded from normal Git history. Store large checkpoints in Google
Drive or a GitHub Release and commit a small manifest here when a model is promoted to `main`.

A manifest should include:

- checkpoint filename and SHA-256;
- source commit and branch;
- exact training command and random seed;
- model and encoder version;
- training data or opponent-pool description;
- fixed-panel and cross-play results;
- external download location.
