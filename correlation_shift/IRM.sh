# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash

# IRM (the main result). These hyperparameters were chosen by random search
# with 50 trials and the following ranges. We picked the values which maximized
# min(train_env0_acc, train_env1_acc, test_acc).  We also chose the steps
# hyperparameter by exhaustive search over [101, 201, 301, 401, 501].
#
# hidden_dim = int(2**np.random.uniform(6, 9))
# l2_regularizer_weight = 10**np.random.uniform(-2, -5)
# lr = 10**np.random.uniform(-2.5, -3.5)
# penalty_anneal_iters = np.random.randint(50, 250)
# penalty_weight = 10**np.random.uniform(2, 6)
echo "IRM:"
python -u main.py \
  --hidden_dim=390 \
  --l2_regularizer_weight=0.00110794568 \
  --lr=0.0004898536566546834 \
  --penalty_anneal_iters=190 \
  --penalty_weight=91257.18613115903 \
  --steps=501 \
  --gpu=0 \
  > IRM.txt
