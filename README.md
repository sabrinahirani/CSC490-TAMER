<div align="center">    
 
# CSC490: Multi-task learning with TAMER model
[![arXiv](https://img.shields.io/badge/arXiv-2408.08578-b31b1b.svg)](https://arxiv.org/abs/2408.08578)

</div>

## Note
This branch (named `multi-task-learning-position-identifier`) is based on
the official github repo of TAMER. It contains all the relevant codes
and files for the experimentation of multi-task learning (based on the 
position forest coding algorithm introduced in PosFormer) and 
depth-weighted structural loss. The base of this branch is the same as
the official TAMER github repo and all the additional work and 
modifications are done by Babur Nawyan.

## Project structure
```bash
├── checkpoints/    # Contains the ckpt file (best model checkpoint), ignore other files
├── config/         # config for TAMER hyperparameter
├── data/
│   └── crohme      # CROHME Dataset
│   └── HME100k      # HME100k Dataset which needs to be downloaded according to the instructions below.
├── eval/             # evaluation scripts
├── logs/
│   └── crohme_training/  # Contains subfolders with hparams.yaml and metrics.csv (record of loss values)
├── results/
│   └── best      # Contains result files for best model, txt files contain the metric values
├── tamer               # model definition folder
├── lightning_logs      # Ignore this folder
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── train.py
```

## Install dependencies   
```bash
cd TAMER
# install project   
conda create -y -n TAMER python=3.7
conda activate TAMER
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
 ```
## Dataset Preparation
We have prepared the CROHME dataset and HME100K dataset in [download link](https://disk.pku.edu.cn/link/AAF10CCC4D539543F68847A9010C607139). After downloading, please extract it to the `data/` folder.

## Training on CROHME Dataset
Next, navigate to TAMER folder and run `train.py`. It may take **32** hours on **1** NVIDIA GeForce RTX 4080 gpu.
```bash
# train TAMER model using 1 gpu on CROHME dataset
python -u train.py --config config/crohme.yaml
```

## Evaluation
Trained TAMER weight checkpoints for the CROHME Dataset have been saved in `checkpoints/epoch=305-step=186659-val_ExpRate=0.6136.ckpt`.

```bash
# For CROHME Dataset
bash eval/eval_crohme.sh checkpoints/epoch=305-step=186659-val_ExpRate=0.6136.ckpt
```
