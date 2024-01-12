# Diffusion Posterior Sampling for Informed Single-channel Dereverberation

This code repository contains the official PyTorch implementation for the paper 

- [*Diffusion Posterior Sampling for Informed Single-Channel Dereverberation*](https://arxiv.org/abs/2306.12286), 2023

Audio examples and supplementary materials are available [on our project page](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/waspaa2023-derevdps.html).

## Installation

- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.

## Pretrained checkpoints

- We provide pretrained checkpoints for the unconditional score model [here](https://drive.google.com/drive/folders/1XG0kc1gzEqWeu75W39QpjHrzAc4FD3HX?usp=sharing)
Usage:
- For resuming training, you can use the `--resume_from_checkpoint` option of `train.py`.
- For evaluating these checkpoints, use the `--ckpt` option of `enhancement.py` (see section **Evaluation** below).

## Training

Training is done by executing `train.py`. A minimal running example with default settings (as in our paper) can be run with

```bash
python train.py --format <your_format>--base_dir <your_base_dir> --gpus 0,
```

where 

- `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). The subdirectory structure depends on `your_format`:
    - `your_format=wsj0`: Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both.
    - Add formats on your own, correpsonding to your data structure

To see all available training options, run `python train.py --help`.
These include options for the backbone DNN, the SDE parameters, the PytorchLightning Trainer usual parameters such as `max_epochs`, `limit_train_batches` and so on.

## Evaluation

To run the model on a test set, run
```bash _enhancement.sh
```
and replace `test_dir` `rir_dir` and `ckpt_score` with the appropriate paths.
Check the paper and different options in `enhancement.py` for different posterior/predictor/corrector/diffusion parameters.

## Data Creation

- In `preprocessing/`, you will find the data generation script used to create all the datasets used in the paper. Minimal example is:

```
    cd preprocessing;
    python3 create_data.py
```

Please check the script for other options

## Citations / References

We kindly ask you to cite our papers in your publication when using any of our research or code:
```bib
@article{lemercier2023derevdps,
  title={Diffusion Posterior Sampling for Informed Single-Channel Dereverberation},
  author={Lemercier, Jean-Marie and Welker, Simon and Gerkmann, Timo},
  journal={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2023}
}
```
