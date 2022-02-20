from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import preprocess_dataset, STNN_Dataset, preprocess_datasets
from model import STNN
from utils import fit_delimiter, elapsed_time_format
from utils import masked_MAE, masked_MAPE
from utils import model_summary, masked_MSE

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


def initialization(args):
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    # Create log dir
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'exp {timestamp}')
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    args.log_dir = log_dir

    # Initialize logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='',
                        filename=os.path.join(log_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    args.log_plotting = os.path.join(log_dir, f"log_plotting.txt")

    # Save hyper-parameters
    logging.info(fit_delimiter('Hyper-parameters', 80))
    for arg in vars(args):
        logging.info(f'{arg}={getattr(args, arg)}')

    return logging


# Tool function
def prepare_dataloaders(args):
    print('Transform Dataset...')
    # Convert data to the sub-spacetime format
    if isinstance(args.data, str):
        train_samples_path, train_targets_path, val_samples_path, val_targets_path, test_samples_path, test_targets_path = \
            preprocess_dataset(args.data, t_in=args.t_history, t_out=args.t_pred,
                               num_nearby_nodes=args.num_nearby_nodes,
                               keep_ratio=args.keep_ratio,
                               debug=args.debug)
    elif isinstance(args.data, list):
        train_samples_path, train_targets_path, val_samples_path, val_targets_path, test_samples_path, test_targets_path = \
            preprocess_datasets(args.data, t_in=args.t_history, t_out=args.t_pred,
                                num_nearby_nodes=args.num_nearby_nodes,
                                keep_ratio=args.keep_ratio,
                                debug=args.debug)
    else:
        raise Exception('Check args.data!')

    if 'test_samples_path' in args and 'test_targets_path' in args:
        test_samples_path = args.test_samples_path
        test_targets_path = args.test_targets_path

    print('Construct DataLoader...')
    # Training set loader
    train_set = STNN_Dataset(train_samples_path, train_targets_path)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=0)

    # Validation set loader
    val_set = STNN_Dataset(val_samples_path, val_targets_path)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, num_workers=0)

    # Test set loader
    test_set = STNN_Dataset(test_samples_path, test_targets_path)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader


def train_batch(model, x, y, optimizer, criterion, device):
    x = x.to(device=device, dtype=torch.float)
    y = y.to(device=device, dtype=torch.float)

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    return loss, output


def train(args, logging, train_dataloader, val_dataloader, test_dataloader):
    # Define model
    model = STNN(args.num_features, args.t_history, args.t_pred,
                 node_num=args.num_nearby_nodes, dropout=args.dropout)
    model = model.to(device=args.device)

    # Warm start
    if 'warmstart' in args:
        checkpoint = torch.load(args.warmstart)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Model summary
    table, total_params = model_summary(model)
    logging.info(f'{table}')
    logging.info(f'Total Trainable Params: {total_params}')

    # Define loss and optimizer
    loss_criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    training_losses = []
    validation_losses = []
    validation_metrics = {'MAEs': [], 'MSEs': [], 'MAPEs': []}
    test_metrics = {'MAEs': [], 'MSEs': [], 'MAPEs': []}
    for epoch in range(args.epochs):
        logging.info(f"------------- Epoch: {epoch:03d} -----------")
        logging.info(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                     f"train batches: {len(train_dataloader)}, "
                     f"val batches: {len(val_dataloader)}, "
                     f"test batches: {len(test_dataloader)}, ")

        # Train
        print('Training')
        batches_train_loss = []
        loop = tqdm(train_dataloader, ncols=100)
        for data, target in loop:
            model.train()
            x = data.to(device=args.device, dtype=torch.float)
            y = target.to(device=args.device, dtype=torch.float)
            loss, out = train_batch(model, x, y, optimizer, loss_criterion, args.device)
            batches_train_loss.append(loss.detach().cpu().numpy())

            loop.set_description(f'Train {epoch + 1}/{args.epochs}')
            loop.set_postfix(loss=np.mean(np.array(batches_train_loss)))
        training_losses.append(np.mean(np.array(batches_train_loss)))

        # Validation
        print('Validation')
        batches_val_loss = []
        batches_val_metrics = {'MAEs': [], 'MSEs': [], 'MAPEs': []}
        with torch.no_grad():
            loop = tqdm(val_dataloader, ncols=100)
            for data, target in loop:
                model.eval()
                x_val = data.to(device=args.device, dtype=torch.float)
                y_val = target.to(device=args.device, dtype=torch.float)
                out = model(x_val)
                val_loss = loss_criterion(out, y_val).to(device="cpu")
                batches_val_loss.append((val_loss.detach().numpy()).item())

                # Metrics
                out_denormalized = out.detach().cpu().numpy().flatten()
                target_denormalized = y_val.detach().cpu().numpy().flatten()
                mae = masked_MAE(out_denormalized, target_denormalized)
                mse = masked_MSE(out_denormalized, target_denormalized)
                mape = masked_MAPE(out_denormalized, target_denormalized)
                if not (np.isnan(mae) or np.isnan(mse) or np.isnan(mape)):
                    batches_val_metrics['MAEs'].append(mae)
                    batches_val_metrics['MSEs'].append(mse)
                    batches_val_metrics['MAPEs'].append(mape)

                loop.set_description(f'Val {epoch + 1}/{args.epochs}')
                move_mae = np.mean(np.array(batches_val_metrics['MAEs']))
                move_mse = np.mean(np.array(batches_val_metrics['MSEs']))
                move_mape = np.mean(np.array(batches_val_metrics['MSEs']))
                loop.set_postfix(MAE=move_mae, MSE=move_mse, MAPE=move_mape)

            assert np.mean(np.array(batches_val_loss)) == sum(batches_val_loss) / len(batches_val_loss)
            epoch_val_loss = np.mean(np.array(batches_val_loss))
            validation_losses.append(epoch_val_loss)
            validation_metrics['MAEs'].append(np.mean(np.array(batches_val_metrics['MAEs'])))
            validation_metrics['MSEs'].append(np.mean(np.array(batches_val_metrics['MSEs'])))
            validation_metrics['MAPEs'].append(np.mean(np.array(batches_val_metrics['MAPEs'])))

            # Save model based on val loss
            if args.save:
                if epoch_val_loss == min(validation_losses):
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                                }, os.path.join(args.log_dir, f'epoch{epoch}_checkpoint.pt'))

        # Testing
        print('Testing')
        batches_test_metrics = {'MAEs': [], 'MSEs': [], 'MAPEs': []}
        with torch.no_grad():
            model.eval()
            loop = tqdm(test_dataloader, ncols=110)
            for data, target in loop:
                x_test = data.to(device=args.device, dtype=torch.float)
                y_test = target.to(device=args.device, dtype=torch.float)
                out = model(x_test)

                # Metrics
                out_denormalized = out.detach().cpu().numpy().flatten()
                target_denormalized = y_test.detach().cpu().numpy().flatten()
                mae = masked_MAE(out_denormalized, target_denormalized)
                mse = masked_MSE(out_denormalized, target_denormalized)
                mape = masked_MAPE(out_denormalized, target_denormalized)
                if not (np.isnan(mae) or np.isnan(mse) or np.isnan(mape)):
                    batches_test_metrics['MAEs'].append(mae)
                    batches_test_metrics['MSEs'].append(mse)
                    batches_test_metrics['MAPEs'].append(mape)

                loop.set_description(f'Test {epoch + 1}/{args.epochs}')
                move_mae = np.mean(np.array(batches_test_metrics['MAEs']))
                move_mse = np.mean(np.array(batches_test_metrics['MSEs']))
                move_mape = np.mean(np.array(batches_test_metrics['MSEs']))
                loop.set_postfix(MAE=move_mae, MSE=move_mse, MAPE=move_mape)

            test_metrics['MAEs'].append(np.mean(np.array(batches_test_metrics['MAEs'])))
            test_metrics['MSEs'].append(np.mean(np.array(batches_test_metrics['MSEs'])))
            test_metrics['MAPEs'].append(np.mean(np.array(batches_test_metrics['MAPEs'])))

        # Print epoch results
        logging.info(f"Pred {args.t_pred} steps - Training loss:   {training_losses[-1]:.8f}")
        logging.info(f"Pred {args.t_pred} steps - Validation loss: {validation_losses[-1]:.8f}")
        logging.info(f"Pred {args.t_pred} steps - Validation MAE:  {validation_metrics['MAEs'][-1]:.4f}")
        logging.info(f"Pred {args.t_pred} steps - Validation RMSE: {np.sqrt(validation_metrics['MSEs'][-1]):.4f}")
        logging.info(f"Pred {args.t_pred} steps - Validation MAPE: {validation_metrics['MAPEs'][-1]:.4f}")
        logging.info(f"Pred {args.t_pred} steps - Test MAE:  {test_metrics['MAEs'][-1]:.4f}")
        logging.info(f"Pred {args.t_pred} steps - Test RMSE: {np.sqrt(test_metrics['MSEs'][-1]):.4f}")
        logging.info(f"Pred {args.t_pred} steps - Test MAPE: {test_metrics['MAPEs'][-1]:.4f}")

        with open(args.log_plotting, 'w') as f:
            print(f"Training loss={training_losses}", file=f)
            print(f"Validation loss={validation_losses}", file=f)
            print(f"Validation MAE={validation_metrics['MAEs']}", file=f)
            print(f"Validation RMSE={np.sqrt(validation_metrics['MSEs'])}", file=f)
            print(f"Validation MAPE={validation_metrics['MAPEs']}", file=f)
            print(f"Test MAE={test_metrics['MAEs']}", file=f)
            print(f"Test RMSE={np.sqrt(test_metrics['MSEs'])}", file=f)
            print(f"Test MAPE={test_metrics['MAPEs']}", file=f)

    return training_losses, validation_losses, validation_metrics, test_metrics


# Main function
def main(args):
    # Initializing
    logging = initialization(args)

    # Prepare train/va/test dataloader
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(args)
    assert next(iter(train_dataloader))[0].shape[-1] == 3

    # Training
    start_time = time.time()
    training_losses, validation_losses, validation_metrics, test_metrics = train(args, logging,
                                                                                 train_dataloader,
                                                                                 val_dataloader,
                                                                                 test_dataloader)
    logging.info(f"Elapsed time: {elapsed_time_format(time.time() - start_time)}")

    # Save summary metrics
    logging.info(fit_delimiter('Performance Summary', 80))
    logging.info(f"Pred {args.t_pred} steps - Top 3 Val MAEs: {np.partition(validation_metrics['MAEs'], 2)[:3]}")
    logging.info(f"Pred {args.t_pred} steps - Top 3 Val RMSEs: {np.partition(validation_metrics['RMSEs'], 2)[:3]}")
    logging.info(f"Pred {args.t_pred} steps - Top 3 Val MAPEs: {np.partition(validation_metrics['MAPEs'], 2)[:3]}")
    logging.info(f"Pred {args.t_pred} steps - Top 3 Test MAEs: {np.partition(test_metrics['MAEs'], 2)[:3]}")
    logging.info(f"Pred {args.t_pred} steps - Top 3 Test RMSEs: {np.partition(test_metrics['RMSEs'], 2)[:3]}")
    logging.info(f"Pred {args.t_pred} steps - Top 3 Test MAPEs: {np.partition(test_metrics['MAPEs'], 2)[:3]}")

    # Save detail metrics
    logging.info(fit_delimiter('Detail Metrics', 80))
    logging.info(f'Val MAEs :  {validation_metrics["MAEs"]}')
    logging.info(f'Val RMSEs:  {validation_metrics["RMSEs"]}')
    logging.info(f'Val MAPEs:  {validation_metrics["MAPEs"]}')
    logging.info(f'Test MAEs : {test_metrics["MAEs"]}')
    logging.info(f'Test RMSEs: {test_metrics["RMSEs"]}')
    logging.info(f'Test MAPEs: {test_metrics["MAPEs"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default=device, help='')
    parser.add_argument('--data', type=str, nargs='+', default='data/METR-LA', help='data path')
    parser.add_argument('--keep_ratio', type=float, default=0.2,
                        help='random sample 20% data from training set to train')
    parser.add_argument('--num_nearby_nodes', type=int, default=15, help='subgraph size')
    parser.add_argument('--num_features', type=int, default=3, help='traffic event: speed, time, location')
    parser.add_argument('--t_history', type=int, default=12, help='T_h')
    parser.add_argument('--t_pred', type=int, default=12, help='T_r')
    parser.add_argument('--target_node', type=int, default=0, help='target node to predict')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=80, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--save', action='store_true', default=True, help='whether save model')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode, faster')
    args = parser.parse_args()

    main(args)
