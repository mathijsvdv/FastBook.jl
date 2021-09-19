# FastAI.jl interfaces

## Training

### High-level

Quickly get started training and finetuning models using already implemented learning methods and callbacks.

{.tight}
- [`methodlearner`](#)
- [`fit!`](#)
- [`fitonecycle!`](#)
- [`finetune!`](#)
- [`BlockMethod`](#)
- callbacks

### Mid-level

{.tight}
- [`Learner`](#)
- [`methodmodel`](#)
- [`methodlossfn`](#)

### Low-level

{.tight}
- [`LearningMethod`](#)
- [`encode`](#)
- [`encodeinput`](#)
- `decodey`

## Datasets

### High-level

Quickly download and load task data containers from the fastai dataset library.

{.tight}
- `load
- [`Datasets.DATASETS`](#)

### Mid-level

Load and transform data containers.

{.tight}
- [`Datasets.datasetpath`](#)
- [`Datasets.FileDataset`](#)
- [`Datasets.TableDataset`](#)
- [`mapobs`](#)
- [`groupobs`](#)
- [`joinobs`](#)
- [`groupobs`](#)

### Low-level

Full control over data containers.

{.tight}
- [`LearnBase.getobs`](#)
- [`LearnBase.nobs`](#)


