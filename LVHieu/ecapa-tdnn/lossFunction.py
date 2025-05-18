
import torch
import torch.nn as nn
import torch.nn.functional as F

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m=0.2, s=30):
        super(AAMsoftmax, self).__init__()
        self.n_class = n_class
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.randn(n_class, 192))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine = torch.matmul(x_norm, w_norm.t())
        phi = cosine - self.m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

