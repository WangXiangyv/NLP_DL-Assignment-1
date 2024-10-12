from src import utils, data_utils
from src.vocabulary import VocabEntry
from src.model import SimpleTokenizer
import torch
from torch import nn

p = nn.Parameter(torch.rand(3, 3))
optimizer = torch.optim.SGD([p], lr=1.0)
# s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
s1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
s2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
s = torch.optim.lr_scheduler.SequentialLR(optimizer, (s1, s2), [3])

for i in range(6):
    print(s.get_last_lr())
    s.step()