import argparse

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import preprocess_dataset, STNN_Dataset
from model import *
from utils import masked_MAE, masked_RMSE, masked_MAPE, masked_MSE

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default=device, help='')
parser.add_argument('--data', type=str, default=r'data/METR-LA', help='data path')
parser.add_argument('--model_path', type=str, default=r'weights/STNN-combined.state.pt', help='model path')
parser.add_argument('--keep_ratio', type=float, default=0.1, help='keep ratio')
parser.add_argument('--num_nearby_nodes', type=int, default=15, help='subgraph size')
parser.add_argument('--t_history', type=int, default=12, help='T_h')
parser.add_argument('--t_pred', type=int, default=12, help='T_r')
parser.add_argument('--num_features', type=int, default=3, help='3 features: speed, time, space')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--mode', type=str, default='client', help='python console use only')
args = parser.parse_args()

print(f'-------------Test {args.data}---------------')

# %% Load model
print('Model loading...')
model = STNN(args.num_features, args.t_history, args.t_pred, node_num=args.num_nearby_nodes)
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)

print('Prepareing dataset...')
_, _, _, _, test_samples_path, test_targets_path = \
    preprocess_dataset(args.data, t_in=args.t_history, t_out=args.t_pred,
                       num_nearby_nodes=args.num_nearby_nodes,
                       train=False, val=False, test=True, target_nodes='all', test_flag=True)
test_set = STNN_Dataset(test_samples_path, test_targets_path)
test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, drop_last=False, num_workers=0)

print('Testing...')
batches_test_metrics_3 = {'MAEs': [], 'MAPEs': [], 'MSEs': []}
batches_test_metrics_6 = {'MAEs': [], 'MAPEs': [], 'MSEs': []}
batches_test_metrics_9 = {'MAEs': [], 'MAPEs': [], 'MSEs': []}
batches_test_metrics_12 = {'MAEs': [], 'MAPEs': [], 'MSEs': []}
with torch.no_grad():
    model.eval()
    loop = tqdm(test_dataloader, ncols=150)
    for data, target in loop:
        x_test = data.to(device=args.device, dtype=torch.float)  # torch.Size([64, 15, 12, 3])
        y_test = target.to(device=args.device, dtype=torch.float)  # torch.Size([64, 12])
        out = model(x_test)
        # Metrics
        out_denormalized_3 = out.detach().cpu().numpy()[:, 0:3].flatten()
        target_denormalized_3 = y_test.detach().cpu().numpy()[:, 0:3].flatten()
        mae_3 = masked_MAE(out_denormalized_3, target_denormalized_3)
        rmse_3 = masked_RMSE(out_denormalized_3, target_denormalized_3)
        mape_3 = masked_MAPE(out_denormalized_3, target_denormalized_3)
        mse_3 = masked_MSE(out_denormalized_3, target_denormalized_3)
        if not (np.isnan(mae_3) or np.isnan(rmse_3) or np.isnan(mape_3)):
            batches_test_metrics_3['MAEs'].append(mae_3)
            batches_test_metrics_3['MAPEs'].append(mape_3)
            batches_test_metrics_3['MSEs'].append(mse_3)

        out_denormalized_6 = out.detach().cpu().numpy()[:, 0:6].flatten()
        target_denormalized_6 = y_test.detach().cpu().numpy()[:, 0:6].flatten()
        mae_6 = masked_MAE(out_denormalized_6, target_denormalized_6)
        rmse_6 = masked_RMSE(out_denormalized_6, target_denormalized_6)
        mape_6 = masked_MAPE(out_denormalized_6, target_denormalized_6)
        mse_6 = masked_MSE(out_denormalized_6, target_denormalized_6)
        if not (np.isnan(mae_6) or np.isnan(rmse_6) or np.isnan(mape_6)):
            batches_test_metrics_6['MAEs'].append(mae_6)
            batches_test_metrics_6['MAPEs'].append(mape_6)
            batches_test_metrics_6['MSEs'].append(mse_6)

        out_denormalized_9 = out.detach().cpu().numpy()[:, 0:9].flatten()
        target_denormalized_9 = y_test.detach().cpu().numpy()[:, 0:9].flatten()
        mae_9 = masked_MAE(out_denormalized_9, target_denormalized_9)
        rmse_9 = masked_RMSE(out_denormalized_9, target_denormalized_9)
        mape_9 = masked_MAPE(out_denormalized_9, target_denormalized_9)
        mse_9 = masked_MSE(out_denormalized_9, target_denormalized_9)
        if not (np.isnan(mae_9) or np.isnan(rmse_9) or np.isnan(mape_9)):
            batches_test_metrics_9['MAEs'].append(mae_9)
            batches_test_metrics_9['MAPEs'].append(mape_9)
            batches_test_metrics_9['MSEs'].append(mse_9)

        out_denormalized_12 = out.detach().cpu().numpy()[:, 0:12].flatten()
        target_denormalized_12 = y_test.detach().cpu().numpy()[:, 0:12].flatten()
        mae_12 = masked_MAE(out_denormalized_12, target_denormalized_12)
        rmse_12 = masked_RMSE(out_denormalized_12, target_denormalized_12)
        mape_12 = masked_MAPE(out_denormalized_12, target_denormalized_12)
        mse_12 = masked_MSE(out_denormalized_12, target_denormalized_12)
        if not (np.isnan(mae_12) or np.isnan(rmse_12) or np.isnan(mape_12)):
            batches_test_metrics_12['MAEs'].append(mae_12)
            batches_test_metrics_12['MAPEs'].append(mape_12)
            batches_test_metrics_12['MSEs'].append(mse_12)

        loop.set_postfix(MAE_03=np.mean(np.array(batches_test_metrics_3['MAEs'])),
                         MSE_03=np.mean(np.array(batches_test_metrics_3['MSEs'])),
                         MAE_06=np.mean(np.array(batches_test_metrics_6['MAEs'])),
                         MSE_06=np.mean(np.array(batches_test_metrics_6['MSEs'])),
                         MAE_12=np.mean(np.array(batches_test_metrics_12['MAEs'])),
                         MSE_12=np.mean(np.array(batches_test_metrics_12['MSEs'])))

print('Predict steps: 3')
print(f"MAE=  {np.mean(batches_test_metrics_3['MAEs']):.6f}")
print(f"RMSE= {np.sqrt(np.mean(batches_test_metrics_3['MSEs'])):.6f}")
print(f"MAPE= {np.mean(batches_test_metrics_3['MAPEs']):.6f}")
print('Predict steps: 6')
print(f"MAE=  {np.mean(batches_test_metrics_6['MAEs']):.6f}")
print(f"RMSE= {np.sqrt(np.mean(batches_test_metrics_6['MSEs'])):.6f}")
print(f"MAPE= {np.mean(batches_test_metrics_6['MAPEs']):.6f}")
print('Predict steps: 9')
print(f"MAE=  {np.mean(batches_test_metrics_9['MAEs']):.6f}")
print(f"RMSE= {np.sqrt(np.mean(batches_test_metrics_9['MSEs'])):.6f}")
print(f"MAPE= {np.mean(batches_test_metrics_9['MAPEs']):.6f}")
print('Predict steps: 12')
print(f"MAE=  {np.mean(batches_test_metrics_12['MAEs']):.6f}")
print(f"RMSE= {np.sqrt(np.mean(batches_test_metrics_12['MSEs'])):.6f}")
print(f"MAPE= {np.mean(batches_test_metrics_12['MAPEs']):.6f}")
