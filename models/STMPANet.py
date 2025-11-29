from models.layers import DilatedInception, LayerNorm, dynamic_graph_constructor, missing_perception
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, 0))

    def forward(self, x):
        # x shape: batch, channels, height, width
        # padding on both ends of the width dimension
        front = x[:, :, :, 0:1].repeat(1, 1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, :, -1:].repeat(1, 1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=3)
        
        # apply average pooling
        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=5, node_dim=40,
                 dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_len=12,
                 out_dim=1, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(Model, self).__init__()
        self.gcn_true = gcn_true #true
        self.buildA_true = buildA_true #true
        self.num_nodes = num_nodes #137
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs_seasonal = nn.ModuleList()
        self.filter_convs_trend = nn.ModuleList()
        self.gate_convs_seasonal = nn.ModuleList()
        self.gate_convs_trend = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs_seasonal = nn.ModuleList()
        self.skip_convs_trend = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gconv3 = nn.ModuleList()
        self.gconv4 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.output_dim=out_dim
        self.seq_length = seq_length
        self.start_conv = nn.Conv2d(in_channels=in_dim,  out_channels=residual_channels,  kernel_size=(1, 1))

        self.dy_gc1 = dynamic_graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                                static_feat=static_feat, dropout=self.dropout, residual_channels=residual_channels)
        self.dy_gc2 = dynamic_graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                                static_feat=static_feat, dropout=self.dropout, residual_channels=residual_channels)

        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            dilationsize=[18, 12, 6]
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs_seasonal.append(DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.filter_convs_trend.append(DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs_seasonal.append(DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs_trend.append(DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs_seasonal.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                              kernel_size=(1, self.seq_length-rf_size_j+1)))
                    self.skip_convs_trend.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                           kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs_seasonal.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                              kernel_size=(1, self.receptive_field-rf_size_j+1)))
                    self.skip_convs_trend.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                           kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(missing_perception(conv_channels, residual_channels, gcn_depth, dropout, propalpha, dilationsize[j-1],
                                                          num_nodes, self.seq_length, out_len, device=device))
                    self.gconv2.append(missing_perception(conv_channels, residual_channels, gcn_depth, dropout, propalpha, dilationsize[j - 1],
                                                          num_nodes, self.seq_length, out_len, device=device))
                    self.gconv3.append(missing_perception(conv_channels, residual_channels, gcn_depth, dropout, propalpha, dilationsize[j - 1],
                                                          num_nodes, self.seq_length, out_len, device=device))
                    self.gconv4.append(missing_perception(conv_channels, residual_channels, gcn_depth, dropout, propalpha, dilationsize[j - 1],
                                                          num_nodes, self.seq_length, out_len, device=device))

                if self.seq_length>self.receptive_field: #
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes,  self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline)) #

                new_dilation *= dilation_exponential #2

        self.layers = layers
        self.end_conv_1 = weight_norm(nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True))
        self.end_conv_2 = weight_norm(nn.Conv2d(in_channels=end_channels, out_channels=out_len*out_dim, kernel_size=(1,1), bias=True))
        if self.seq_length > self.receptive_field:
            self.skip0 = weight_norm(nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True))
            self.skipE = weight_norm(nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True))
        else:
            self.skip0 = weight_norm(nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True))
            self.skipE = weight_norm(nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True))

        self.idx = torch.arange(self.num_nodes).to(device)
        self.decomp_multi = series_decomp(7)

    
    def partial_conv(self, layer_index, x, mask, _type="seasonal"):
        if _type == "seasonal":
            filter, mask_filter = self.filter_convs_seasonal[layer_index](x, mask)
            filter = torch.tanh(filter)
            gate,mask_gate = self.gate_convs_seasonal[layer_index](x, mask)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs_seasonal[layer_index](s)
        elif _type == "trend":
            filter, mask_filter = self.filter_convs_trend[layer_index](x, mask)
            filter = torch.tanh(filter)
            gate,mask_gate = self.gate_convs_trend[layer_index](x, mask)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs_trend[layer_index](s)
        else:
            return
        return s, mask_filter, x


    def forward(self, input, mask, k, idx=None):#tx,id
        input = input.transpose(1, 3)
        mask = mask.transpose(1, 3).float()
        input = input * mask
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field - self.seq_length, 0, 0, 0))

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    self.adp1 = self.dy_gc1(x, self.idx)    # dynamic graph
                else:
                    self.adp1 = self.dy_gc1(x, idx)
            else:
                adp = self.predefined_A

        for i in range(self.layers):
            residual = x
            seasonal_init_enc, trend = self.decomp_multi(x)

            # temporal
            s1, mask_filter_1, seasonal_init_enc = self.partial_conv(i, seasonal_init_enc, mask, _type="seasonal")
            s2, mask_filter_2, trend = self.partial_conv(i, trend, mask, _type="trend")
            skip = s1 + s2 + skip

            #  spatial
            if self.gcn_true:
                # seasonal
                state1, mask1 = self.gconv1[i](seasonal_init_enc, self.adp1[0], mask_filter_1, k, flag=0)
                # trend
                state3, mask3 = self.gconv3[i](trend, self.adp1[1], mask_filter_2, k, flag=0)

                x = state1 + state3
                mask = (mask1 + mask3) / 2
            else:
                x = self.residual_convs[i](seasonal_init_enc + trend)
                mask = (mask_filter_1 + mask_filter_2) / 2

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        B, T, N, D = x.shape
        x = x.reshape(B,-1,self.output_dim,N)
        x = x.permute(0,1,3,2)
        return x

