# Revisiting Contrastive Learning in Out-of-Distribution Generalization

## Introduction
This is a PyTorch implementation of our paper [Revisiting Contrastive Learning in Out-of-Distribution Generalization](https://).

![framework](./figures/framework.png)

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

Update the "--image_path" in /diversity_shift/ERM_CID.sh according to your path.

### Run
```
cd diversity_shift

sh ERM_CID.sh
```


## Reference
Our implementation references the codes in the following repositories:
* [InvariantRiskMinimization](https://github.com/facebookresearch/InvariantRiskMinimization)
* [JigenDG](https://github.com/fmcarlucci/JigenDG)
