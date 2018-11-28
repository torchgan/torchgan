import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dims):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(input_dims, input_dims//8, 1)
        self.key = nn.Conv2d(input_dims, input_dims//8, 1)
        self.value = nn.Conv2d(input_dims, input_dims, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        dims = (x.size(0), -1, x.size(2) * x.size(3))
        out_query = self.query(x).view(dims)
        out_key = self.key(x).view(dims).permute(0, 2, 1)
        attn = F.softmax(torch.bmm(out_key, out_query), dim=-1)
        out_value = self.value(x).view(dims)

        out = self.gamma * torch.bmm(out_value, attn) + x
        return out, attn
