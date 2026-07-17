# Branch workflow

This repository uses three long-lived branches.

## `main`

Stable submission and model code. Merge a training change here only after its checkpoint has been
evaluated against the fixed panel and the relevant tests pass. Large checkpoints are stored outside
normal Git history; record their filename, hash, training command, and evaluation result in a model
manifest when promoting a model.

## `local`

Mac and MPS development. Use small rollout counts for feature, simulator, and checkpoint smoke tests.
Core code changes should be merged back to `main` after validation instead of living only here.

## `colab`

CUDA/T4 training. This branch contains the Colab bootstrap and long-running training command. Pull or
merge `main` before starting a new experiment, then merge portable code improvements back to `main`.

## Suggested promotion flow

```text
main -> local -> main -> colab -> main
```

1. Branch or merge from the latest `main` into `local` and run fast tests.
2. Merge portable implementation changes into `main`.
3. Bring the tested `main` commit into `colab` and run the full CUDA experiment.
4. Evaluate the resulting checkpoint, store it in Drive or a Release, and promote its configuration
   and manifest to `main`.

Never commit Kaggle credentials, the restricted official simulator, raw competition downloads,
generated datasets, or checkpoint binaries.
