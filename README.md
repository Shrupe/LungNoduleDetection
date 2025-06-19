Final Project
Used Dataset: https://luna16.grand-challenge.org/

https://www.kaggle.com/code/gzuidhof/full-preprocessing-tutorial

Preprocess .mhd > Create processed .npy files > Positive patches: Extract nodule centered 3d (32, 32, 32) patches (create one with augmentation), Negative patches: Extract 3d patches with no nodule > Split the data and train

!!! Model in this project searches nodule in the center of the given patch.

Results:
  [Train] Loss: 0.2130 | Acc: 0.9204 | AUC: 0.9677
  [Val]   Loss: 0.2167 | Acc: 0.9241 | AUC: 0.9642 | F1: 0.9198 | P: 0.9741 | R: 0.8713 | T: 0.60
