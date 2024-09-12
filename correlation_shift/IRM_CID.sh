
echo "IRM_CID (ours):"
python -u main_CID.py \
  --hidden_dim=390 \
  --l2_regularizer_weight=0.00110794568 \
  --lr=0.0004898536566546834 \
  --penalty_anneal_iters=190 \
  --penalty_weight=91257.18613115903 \
  --steps=501 \
  --beta=10 \
  --temperature=0.35 \
  --gpu=0 \
  > IRM_CID_beta10_t0.35.txt
