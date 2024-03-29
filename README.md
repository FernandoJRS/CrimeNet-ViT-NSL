# CrimeNet: Neural Structured Learning using Vision Transformer for Violence Detection 

## CrimeNet

CrimeNet is a Visit Transformer (ViT)-based deep learning model that employs neural structured learning with adversarial regularization for violence detection. In this repository we show some tests carried out with videos of violence in real environments.

## Performances CrimeNet

This section shows some performance of the CrimeNet model.

[![Watch the video](https://github.com/FernandoJRS/CrimeNet-ViT-NSL/blob/main/video_01.png)](https://drive.google.com/file/d/1Q1teUnISw3N5-Q4rHwRZ82qV08-11ObX/view?usp=sharing)

[![Watch the video](https://github.com/FernandoJRS/CrimeNet-ViT-NSL/blob/main/video_02.png)](https://drive.google.com/file/d/1rCyn0UtEpiFow1Z6-BoS6O6-wutN-O_m/view?usp=sharing)

[![Watch the video](https://github.com/FernandoJRS/CrimeNet-ViT-NSL/blob/main/video_03.png)](https://drive.google.com/file/d/1NzTYrRNsa1Yuat5HDLhiY3OjkJmMItM5/view?usp=sharing)

## Code

### UBI Fights

This repository provides the evaluation code and the pre-trained CrimeNet model for the UBI-Fights dataset. It is located in the UBI_Fights directory. In it you can find the Jupyter evaluation notebook EvaluateCrimeNet, the script with the ViT model architecture and in the subdirectory Results/logs/checkpoint/, in the file check.txt you can find the link to download the pre-trained model. The training workbook is also provided in CrimeNetTraining, as well as a Tensorboard with all the related graphs and data whose checkpoint is located in Results/logs/checkpoint/ in the tensorboard file.

[UBI_Fights dataset](http://socia-lab.di.ubi.pt/EventDetection/)

[UBI_Fights ViT](https://github.com/FernandoJRS/CrimeNet-ViT-NSL/blob/main/UBI_Fights/ViT.py)

[UBI_Fights CrimeNetTraining](https://github.com/FernandoJRS/CrimeNet-ViT-NSL/blob/main/UBI_Fights/CrimeNetTraining.ipynb)

[UBI_Fights EvaluateCrimeNet](https://github.com/FernandoJRS/CrimeNet-ViT-NSL/blob/main/UBI_Fights/EvaluateCrimeNet.ipynb)

The AUC PR and AUC ROC plots for CrimeNet are shown below.

![UBI AUC ROC](UBI_Fights/figures/auc_roc.png?raw=True "UBI AUC ROC") | ![UBI AUC PR](UBI_Fights/figures/auc_pr.png?raw=True "UBI AUC PR")
