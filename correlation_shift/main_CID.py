# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--beta', default=10, type=float, help='weight on class_aware_contrastive loss')
parser.add_argument('--temperature', default=0.35, type=float, help='Temperature used in class_aware_contrastive loss')
parser.add_argument('--gpu', default=1, type=int, help='the GPU ID')
flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
  print("Restart", restart)
  device = torch.device('cuda', flags.gpu)
  # Load MNIST, make train/val splits, and shuffle train set examples

  mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
  mnist_train = (mnist.data[:50000], mnist.targets[:50000])
  mnist_val = (mnist.data[50000:], mnist.targets[50000:])

  rng_state = np.random.get_state()
  np.random.shuffle(mnist_train[0].numpy())
  np.random.set_state(rng_state)
  np.random.shuffle(mnist_train[1].numpy())

  # Build environments

  def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
      return (torch.rand(size) < p).float()
    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.).to(device),
      'labels': labels[:, None].to(device)
    }

  # import pdb; pdb.set_trace()
  envs = [
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),  # mnist_train[0][::2]: [25000,28,28] 
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9)  # mnist_val[0]: [10000,28,28] 
  ]

  # Define and instantiate the model

  class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      if flags.grayscale_model:
        self.lin1 = nn.Linear(14 * 14, flags.hidden_dim)
      else:
        self.lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
      self.lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
      self.lin3 = nn.Linear(flags.hidden_dim, 1)
      for lin in [self.lin1, self.lin2, self.lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(self.lin1, nn.ReLU(True), self.lin2, nn.ReLU(True))
      
    def forward(self, input):
      if flags.grayscale_model:
        out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
      else:
        out = input.view(input.shape[0], 2 * 14 * 14)
      features = self._main(out)
      out = self.lin3(features)
      return features, out

  mlp = MLP().to(device)

  # Define loss function helpers

  def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

  def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

  def penalty(logits, y):
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

  def class_contras_loss(z, class_l):
        # import pdb; pdb.set_trace()
        z_norm = nn.functional.normalize(z, dim=-1)
        # the consine similarity between all pairs 
        sim_matrix = torch.exp(torch.mm(z_norm, z_norm.t().contiguous()) / flags.temperature) # [Batch, Batch]
        mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool() # [Batch, Batch]
        sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)  # [Batch, Batch-1]

        # the consine similarity between positive pairs 
        l = class_l.view(sim_matrix.shape[0], -1)
        pos_mask = torch.eq(l, l.t().contiguous()) # [Batch, Batch]
        pos_mask = pos_mask.masked_select(mask).view(sim_matrix.shape[0], -1).float()  # [Batch, Batch-1]
        pos_sim_matrix = sim_matrix * pos_mask  # [Batch, Batch-1]

        # compute loss
        loss = (- torch.log(pos_sim_matrix[pos_mask.sum(dim=-1)>0.9].sum(dim=-1) / sim_matrix[pos_mask.sum(dim=-1)>0.9].sum(dim=-1))).mean()
        # print(loss)
        return loss


  # Train loop

  def pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

  params = list(mlp.parameters())
  optimizer = optim.Adam(params, lr=flags.lr)

  pretty_print('step', 'train nll', 'train acc', 'train penalty', 'CID_Loss', 'test acc')

  for step in range(flags.steps):
    for env in envs:
      # import pdb; pdb.set_trace()
      features, logits = mlp(env['images'])

      env['nll'] = mean_nll(logits, env['labels'])
      env['acc'] = mean_accuracy(logits, env['labels'])
      env['penalty'] = penalty(logits, env['labels'])
      env['CID'] = class_contras_loss(features, env['labels'])

    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
    class_aware_contrastive_loss = torch.stack([envs[0]['CID'], envs[1]['CID']]).mean()

    weight_norm = torch.tensor(0.).to(device)
    for w in mlp.parameters():
      weight_norm += w.norm().pow(2)

    loss = train_nll + flags.beta * class_aware_contrastive_loss
    loss += flags.l2_regularizer_weight * weight_norm
    penalty_weight = (flags.penalty_weight 
        if step >= flags.penalty_anneal_iters else 1.0)
    loss += penalty_weight * train_penalty
    if penalty_weight > 1.0:
      # Rescale the entire loss to keep gradients in a reasonable range
      loss /= penalty_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    test_acc = envs[2]['acc']
    if step % 100 == 0:
      pretty_print(
        np.int32(step),
        train_nll.detach().cpu().numpy(),
        train_acc.detach().cpu().numpy(),
        train_penalty.detach().cpu().numpy(),
        class_aware_contrastive_loss.detach().cpu().numpy(),
        test_acc.detach().cpu().numpy()
      )

  final_train_accs.append(train_acc.detach().cpu().numpy())
  final_test_accs.append(test_acc.detach().cpu().numpy())
  print('Final train acc (mean/std across restarts so far):')
  print(np.mean(final_train_accs), np.std(final_train_accs))
  print('Final test acc (mean/std across restarts so far):')
  print(np.mean(final_test_accs), np.std(final_test_accs))
