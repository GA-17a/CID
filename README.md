# Revisiting Contrastive Learning in Out-of-Distribution Generalization

## Introduction
This is a PyTorch implementation of our paper [Revisiting Contrastive Learning in Out-of-Distribution Generalization](https://).

<div align="center">
  <img src="https://github.com/GA-17a/CID/tree/main/figures/framework.png">
</div>
<!-- <p align="center">
  Figure 1: Framework of different methods. (a) The baseline model is trained only with the classification task. (b) The comparative model is trained with both the classification task and the self-supervised instance discrimination task. (c) Our model is trained with both the classification task and the class-wise instance discrimination task.
</p> -->

## Correlation Shift
```
cd correlation_shift

# ERM
sh ERM.sh

# IRM
sh IRM.sh

# Ours
sh IRM_CID.sh
```

## Diversity Shift

### Dataset Preparation
Download the [PACS dataset](https://arxiv.org/abs/1710.03077) from: [download_pacs](https://wjdcloud.blob.core.windows.net/dataset/PACS.zip).

Update the "--image_path" in /diversity_shift/ERM_CID.sh to your path.

### Run
```
cd diversity_shift

sh ERM_CID.sh
```


## Reference
Our implementation references the codes in the following repositories:
* [InvariantRiskMinimization](https://github.com/facebookresearch/InvariantRiskMinimization)
* [JigenDG](https://github.com/fmcarlucci/JigenDG)
