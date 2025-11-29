from tqdm import tqdm
import torch
import numpy as np
import argparse
import json
from models.STMPANet import Model
from data.GenerateDataset import loaddataset

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--task', default='prediction', type=str)
parser.add_argument("--adj-threshold", type=float, default=0.1)
parser.add_argument('--dataset', default='BeijingAir')
parser.add_argument('--val_ratio', default=0.2)
parser.add_argument('--test_ratio', default=0.2)
parser.add_argument('--column_wise', default=False)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument("--model-name", type=str, default='spin')
parser.add_argument("--dataset-name", type=str, default='air36')
parser.add_argument('--fc_dropout', default=0.2, type=float)
parser.add_argument('--head_dropout', default=0, type=float)
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')

parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--version', type=str, default='Fourier')
parser.add_argument('--mode_select', type=str, default='random')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh')
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--output_dim', default=1, type=int)
parser.add_argument('--embed_dim', default=512, type=int)
parser.add_argument('--rnn_units', default=64, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--cheb_k', default=2, type=int)
parser.add_argument('--default_graph', type=bool, default=True)

parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')
parser.add_argument("--config_filename", type=str, default='')
parser.add_argument("--config", type=str, default='imputation/spin.yaml')
parser.add_argument('--output_attention', type=bool, default=False)

parser.add_argument('--val_len', type=float, default=0.2)
parser.add_argument('--test_len', type=float, default=0.1)
parser.add_argument('--mask_ratio', type=float, default=0.2)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--l2-reg', type=float, default=0.)
parser.add_argument('--batch-inference', type=int, default=32)
parser.add_argument('--split-batch-in', type=int, default=1)
parser.add_argument('--grad-clip-val', type=float, default=5.)
parser.add_argument('--loss-fn', type=str, default='l1_loss')
parser.add_argument('--lr-scheduler', type=str, default=None)
parser.add_argument('--seq_len', default=24, type=int)  # 96
parser.add_argument('--history_len', default=24, type=int)  # 96
parser.add_argument('--label_len', default=12, type=int)  # 48
parser.add_argument('--pred_len', default=24, type=int)
parser.add_argument('--horizon', default=24, type=int)
parser.add_argument('--delay', default=0, type=int)
parser.add_argument('--stride', default=1, type=int)
parser.add_argument('--window_lag', default=1, type=int)
parser.add_argument('--horizon_lag', default=1, type=int)

parser.add_argument('--gcn_depth', default=1, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--node_dim', default=32, type=int)
parser.add_argument('--conv_channels', default=16, type=int)
parser.add_argument('--residual_channels', default=32, type=int)
parser.add_argument('--skip_channels', default=32, type=int)
parser.add_argument('--end_channels', default=64, type=int)
parser.add_argument('--layers', default=3, type=int)
parser.add_argument('--propalpha', default=0.05, type=float)
parser.add_argument('--tanhalpha', default=3.0, type=float)
parser.add_argument('--exp_name', default=None, type=str)
parser.add_argument('--in_dim', default=1, type=int)







args = parser.parse_args()



if args.dataset == 'PEMS':
    node_number = 325
    args.num_nodes = 325
    args.enc_in = 325
    args.dec_in = 325
    args.c_out = 325
elif args.dataset == 'BeijingAir':
    node_number = 36
    args.num_nodes = 36
    args.enc_in = 36
    args.dec_in = 36
    args.c_out = 36
elif args.dataset == 'Weather':
    node_number = 21
    args.num_nodes = 21
    args.enc_in = 21
    args.dec_in = 21
    args.c_out = 21


def MAPE_np(pred, true, mask_value=0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), (true)))) * 100


def RMSE_np(pred, true, mask_value=0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred - true)))
    return RMSE


def test(model):
    loss = 0.0
    labels = []
    preds = []

    train_dataloader, val_dataloader, test_dataloader, scaler = loaddataset(args.history_len, args.pred_len,
                                                                            args.mask_ratio, args.dataset,
                                                                            args.batch_size,
                                                                            args.val_len, args.test_len)
    model.eval()
    k = 0
    with torch.no_grad():
        for i, (x, y, mask, target_mask) in enumerate(tqdm(test_dataloader)):
            x, y, mask, target_mask = x.to(args.device), y.to(args.device), mask.to(args.device), target_mask.to(
                args.device)
            x_hat = model(x, mask, k)
            k = k + 1
            x_hat = scaler.inverse_transform(x_hat)

            y = scaler.inverse_transform(y)
            preds.append(x_hat.squeeze())
            labels.append(y.squeeze())
            losses = torch.sum(torch.abs(x_hat - y) * (target_mask)) / torch.sum(target_mask)
            loss += losses

        labels = torch.cat(labels, dim=0).cpu().numpy()
        preds = torch.cat(preds, dim=0).cpu().numpy()

        print("mask loss: ", loss / len(test_dataloader))
        loss = np.mean(np.abs(labels.squeeze() - preds.squeeze()))
        RMSE = RMSE_np(preds.squeeze(), labels.squeeze())
        MAPE = MAPE_np(preds.squeeze(), labels.squeeze())
        print("loss: %.2f,  RMSE: %.2f, MAPE: %.2f" % (loss, RMSE, MAPE))

    return loss


def run():
    model = Model(False, False, gcn_depth=args.gcn_depth, num_nodes=node_number, device=args.device, predefined_A=False,
                  dropout=args.dropout, subgraph_size=10,
                  node_dim=args.node_dim,
                  dilation_exponential=1,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.seq_len, in_dim=args.in_dim, out_len=args.pred_len, out_dim=1,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True).to(args.device)

    output_path = './output/' + args.dataset + '_miss' + str(args.mask_ratio)
    best_path = output_path + f"/{args.exp_name}_best.pth"
    state_dict = torch.load(best_path)
    model.load_state_dict(state_dict['model_state_dict'])
    loss = test(model)

    print('loss:', loss)


def load_args_from_json(json_path, args):
    with open(json_path, 'r') as f:
        json_args = json.load(f)
    for key, value in json_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args



if __name__ == '__main__':
    json_path = f'./log_dir/{args.dataset}_miss{args.mask_ratio}_{args.exp_name}.json'
    args = load_args_from_json(json_path, args)

    print(args)
    run()
