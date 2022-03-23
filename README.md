# STNN
This is the PyTorch implementation of paper *["Space Meets Time: Local Spacetime Neural Network For Trafﬁc Flow Forecasting"](https://arxiv.org/pdf/2109.05225)*.

## Installation
```
pip install -r requirements.txt
```

## Requirements
- pytorch (1.7 or later)
- numpy
- prettytable
- tqdm


## Train
Before train, unzip dataset to `data/METR-LA`, `data/PeMS-Bay`

```
# Train on PeMS-Bay
python train.py --data data/PeMS-Bay --t_history 12 --t_pred 12 --keep_ratio 0.2
```

## Test
This single model can be used in both METR-LA and PeMS-Bay traffic prediction
```
python test.py --data data/METR-LA --model weights/STNN-combined.state.pt
python test.py --data data/PeMS-Bay --model weights/STNN-combined.state.pt
```

## Citation
```
@article{yang2021space,
  title={Space Meets Time: Local Spacetime Neural Network For Traffic Flow Forecasting},
  author={Yang, Song and Liu, Jiamou and Zhao, Kaiqi},
  journal={arXiv preprint arXiv:2109.05225},
  year={2021}
}
```
