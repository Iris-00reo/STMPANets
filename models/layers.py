from torch.nn import init
import numbers
import numpy as np
import scipy.sparse as sp
from torch.nn.utils import weight_norm
import math
import torch
import torch.nn.functional as F
from torch import nn


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class nconv_dynamic(nn.Module):
    def __init__(self):
        super(nconv_dynamic, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bcwl,bvnw->bcnl', (x, A))
        return x.contiguous()


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_o = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, C = x.size()
        Q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.W_o(attn_output)
        return output


class enhance_mask(nn.Module):
    def __init__(self, inputsize, num_nodes, input_len, pred_len, device, residual_channels):
        super(enhance_mask, self).__init__()
        self.pred_len = pred_len
        self.input_len = input_len
        self.num_nodes = num_nodes
        self.inputsize = inputsize
        self.device = device
        self.residual_channels = residual_channels
        self.beta = 0.02
        self.softmax = nn.Softmax(dim=-1)
        # mask enhance
        self.attn = Attention(embed_dim=self.num_nodes * self.residual_channels, num_heads=4).to(self.device)

    def forward(self, x, A, mask, k):
        B, C1, N, L = mask.shape
        mask_updated = mask

        # node feature similarity
        S = self.softmax(torch.matmul(x, x.transpose(-1, -2)))  # torch.Size([B, C, N, N])
        S = torch.mean(S, dim=1, keepdim=True).expand(-1, C1, -1, -1)   # torch.Size([B, C1, N, N])

        # update adjacency matrix by using enhanced mask
        mask_weight = S * torch.matmul(mask_updated, mask_updated.transpose(-1, -2))
        mask_weight = torch.sigmoid(mask_weight / torch.sqrt(torch.tensor(10.0)))

        A = torch.mean(A, dim=1, keepdim=True).expand(-1, C1, -1, -1)   # # torch.Size([B, C1, N, N])
        A = A + self.beta * mask_weight

        # update mask matrix by aggregating topk A neighbor nodes
        _, topk_A = torch.topk(A, k=5, dim=-1, largest=True)
        topk_A = topk_A.unsqueeze(-1).expand(-1, C1, -1, -1, self.inputsize)
        mask_updated = mask_updated.unsqueeze(2).expand(-1, C1, self.num_nodes, -1, -1)
        mask_select = mask_updated.gather(3, topk_A)
        mask_updated, _ = torch.max(mask_select, dim=-2)

        return A.contiguous(), mask_updated.contiguous()


class missing_perception(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha, inputsize,
                 num_nodes, input_len, pred_len, device):
        super(missing_perception, self).__init__()

        self.enhance_mask = enhance_mask(inputsize, num_nodes, input_len, pred_len, device, c_in)
        self.nconv_dynamic = nconv_dynamic()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.device = device


    def forward(self, x, adj, mask, k, flag=0):
        h = x
        out = [h]

        for i in range(self.gdep):  # 2
            A, mask = self.enhance_mask(h, adj, mask, k)    # A: torch.Size([B, c1, N, N]) mask: torch.Size([B, c1, N, L])
            state = self.nconv_dynamic(h, A)    # torch.Size([B, C, N, L])

            h = self.alpha * x + (1 - self.alpha) * state
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)

        return ho, mask


class nconv_gwnet(nn.Module):
    def __init__(self):
        super(nconv_gwnet, self).__init__()

    def forward(self, x, A):
        # print("x.size is {}, A.size is {}".format(x.size(), A.size()))
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class gcn_gwnet(nn.Module):
    def __init__(self, c_in=32, c_out=32, dropout=0.3, support_len=3, order=2):
        super(gcn_gwnet, self).__init__()
        self.nconv = nconv_gwnet()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]   # x.size(): (N, c_in, D, L)
        for a in support:   # 2 direction adjacency matrix
            x1 = self.nconv(x, a)    # x1: (N, c_in, D, L)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)    # (N, c_in, D, L)
        h = self.mlp(h)     # [N, c_out, D, L]
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class dynamic_graph_constructor(nn.Module):
    def __init__(self, nnodes, k, node_dim, device, alpha=3, static_feat=None, dropout=0.3, residual_channels=32):
        super(dynamic_graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.node_dim = node_dim
        self.device = device
        self.alpha = alpha
        self.static_feat = static_feat
        self.dropout = dropout
        self.residual_channels = residual_channels
        if static_feat is not None:
            xd = self.static_feat.shape[1]
            self.lin1 = nn.Linear(xd, self.node_dim)
            self.lin2 = nn.Linear(xd, self.node_dim)
        else:
            self.emb1 = nn.Embedding(self.nnodes, self.node_dim)
            self.emb2 = nn.Embedding(self.nnodes, self.node_dim)
            self.lin1 = nn.Linear(self.node_dim, self.node_dim)
            self.lin2 = nn.Linear(self.node_dim, self.node_dim)

        self.dy_filter1 = gcn_gwnet(c_in=self.residual_channels, c_out=self.residual_channels, dropout=self.dropout, support_len=1)
        self.dy_filter2 = gcn_gwnet(c_in=self.residual_channels, c_out=self.residual_channels, dropout=self.dropout, support_len=1)

    def asym_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).astype(np.float32).todense()

    def prep_adj(self, adj):
        adj = adj + torch.eye(self.nnodes).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return adj

    def forward(self, x, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        # Gaussian kernel function
        G = torch.exp(-torch.cdist(nodevec1, nodevec2, p=2) ** 2 / (2 * self.alpha ** 2))   # [num_nodes, nums_nodes]
        adj_mx = [self.asym_adj(G.cpu().detach().numpy()), self.asym_adj(np.transpose(G.cpu().detach().numpy()))]
        adj_mx = [torch.tensor(i).to(self.device) for i in adj_mx]

        filter1 = self.dy_filter1(x, [adj_mx[0]])  # (N, residual_channels, D, L)
        filter2 = self.dy_filter2(x, [adj_mx[1]])  # (N, residual_channels, D, L)
        filter1 = filter1.permute((0, 3, 2, 1)).contiguous()  # (N, L, D, residual_channels)
        filter2 = filter2.permute((0, 3, 2, 1)).contiguous()  # (N, L, D, residual_channels)
        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))  # (N, L, D, residual_channels)
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))
        a = torch.matmul(nodevec1, nodevec2.transpose(2, 3)) - torch.matmul(nodevec2, nodevec1.transpose(2, 3))
        adj = F.relu(torch.tanh(self.alpha * a))    # (B, L, D, D)

        mask = torch.zeros(adj.size(0), adj.size(1), adj.size(2), adj.size(3)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(20, -1)
        mask.scatter_(-1, t1, s1.fill_(1))
        adj = adj * mask

        adj = self.prep_adj(adj)
        adjT = self.prep_adj(adj.transpose(2, 3))
        return [adj, adjT]


    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2, alpha=0.5):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.mconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        self.alpha = alpha
        
        for kern in self.kernel_set:
            self.tconv.append(weight_norm(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor))))
            self.mconv.append(nn.Conv2d(1, 1, (1, kern), dilation=(1, dilation_factor)))


    def init_weight(self):
        for name, module in self.named_modules():
            if 'mconv' in name and isinstance(module, nn.Conv2d):
                in_size, out_size, h, w = module.weight.shape
                module.weight = nn.Parameter(torch.ones(in_size, out_size, h, w), requires_grad=False)

    def forward(self, input, mask):
        x = []
        mask_list = []
        mask = mask[:, :1, :, :]

        for i in range(len(self.kernel_set)):
            feature_x = self.tconv[i](input)
            mask_weight = self.mconv[i](mask)
            feature_x = feature_x * mask_weight
            x.append(feature_x)
            mask_list.append(mask_weight)

        min_width = min([feat.size(3) for feat in x])

        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., :min_width]
            mask_list[i] = mask_list[i][..., :min_width]

        x = torch.cat(x, dim=1)
        mask = torch.cat(mask_list, dim=1)
        
        # Calculate the valid ratio
        mask_count = torch.sum(mask, dim=1, keepdim=True)  # torch.Size([32, 1, 207, 18])
        mask_size = mask.size(3)
        valid_ratio = mask_count / mask_size * 4
        new_mask = (valid_ratio > self.alpha).float()
        return x, new_mask

