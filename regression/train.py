import logging
import os
import yaml
import argparse
import torch
from torch_geometric.loader import DataLoader as GraphLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from data.datasets import TriangleFeatureDataset, TriangleFeatureGraphDataset
from src.models import GraphTriangleEncoder, TriangleFeatureExtractor
from models import UNetTriangleModelRegression
from src.utils import save_checkpoint, setup_directories, save_config,initialize, save_results_to_csv
from utils import get_loss_fn

def train(model,
          train_loader,
          val_loader,
          optimizer,
          loss_fn,
          num_epochs,
          log_dir,
          out_dir,
          device,
          scheduler = None,
          start_epoch = 0,
          features=None):

    val_loss = float('inf')
    logger = SummaryWriter(log_dir)
    logging.info("Start training...")

    # Progress bar
    prog_bar = tqdm(initial = start_epoch, total=start_epoch+num_epochs, desc="Training")

    for epoch in range(start_epoch, start_epoch+num_epochs):
        model.train()
        epoch_loss = []

        for batch in train_loader:

            x, y = batch['x'].to(device), batch['y'].to(device)
            additional_inputs = []
            if x.ndim > 3:
                x = x.squeeze(0)
                y = y.squeeze(0)
            if batch.get('edge_index', None) is not None:
                edge_index = batch['edge_index'].to(device)
                additional_inputs.append(edge_index)
            if batch.get('edge_attr', None) is not None:
                edge_attr = batch['edge_attr'].to(device)
                additional_inputs.append(edge_attr)
            preds = model(x,*additional_inputs)
            loss = loss_fn(preds, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        mean_epoch_loss = sum(epoch_loss) / len(epoch_loss)

        # Validation step
        eval_dict = evaluate(model, val_loader,loss_fn,features, device)

        # Save best model
        if eval_dict['loss'] < val_loss:
            val_loss = eval_dict['loss']
            save_checkpoint(epoch, model, optimizer, os.path.join(out_dir,'best_model.pkl'), scheduler)

        # Logging
        logger.add_scalar('Train/loss', mean_epoch_loss, epoch)
        for key, value in eval_dict.items():
            if key != 'residuals':
                logger.add_scalar(f'Val/{key}', value, epoch)
        if scheduler:
            logger.add_scalar('Train/lr', scheduler.get_last_lr()[0], epoch)
            scheduler.step()

        prog_bar.set_description(f"Train Loss: {mean_epoch_loss:.4f} Validation Loss: {eval_dict['loss']:.4f}")
        prog_bar.update()

    # Save latest model checkpoint
    save_checkpoint(epoch, model, optimizer, os.path.join(out_dir, 'last_model.pkl'), scheduler)
    logger.close()
    
def evaluate(model,data_loader,loss_fn,features, device):

    model.eval()
    mean_loss = 0.
    mean_val_error = np.zeros(7)
    N = len(data_loader)
    y_squared = np.zeros(7)
    ssd = np.zeros(7)
    all_residuals = {feature: [] for feature in features}
    with (torch.no_grad()):
        for batch in data_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            additional_inputs = []
            if x.ndim > 3:
                x = x.squeeze(0)
                y = y.squeeze(0)
            if batch.get('edge_index', None) is not None:
                edge_index = batch['edge_index'].to(device)
                additional_inputs.append(edge_index)
            if batch.get('edge_attr', None) is not None:
                edge_attr = batch['edge_attr'].to(device)
                additional_inputs.append(edge_attr)
            preds = model(x,*additional_inputs)
            loss = loss_fn(preds, y)
            val_error =  torch.sqrt(torch.mean((preds-y)**2,dim=(0,1))) #sum over first 2 dimensions, assumes batch dimension exists
            ssd += torch.sum((preds-y)**2,dim=(0,1)).detach().cpu().numpy()
            y_squared += torch.sum(y**2,dim=(0,1)).detach().cpu().numpy()

            # Collect residuals for violin plots
            residuals = (preds - y).squeeze().detach().cpu().numpy()
            for i, feature in enumerate(features):
                all_residuals[feature].extend(residuals[:, i])

            mean_loss += loss.item()
            mean_val_error += val_error.detach().cpu().numpy()

        mean_loss /= N
        mean_val_error /= N
        rel_error = np.sqrt(ssd / y_squared)
    eval_dict = {'loss': mean_loss}
    # Add RMSE and relative error metrics
    for i, col in enumerate(features):
        eval_dict[f"{col}_rmse"] = mean_val_error[i]
        eval_dict[f"{col}_rel_error"] = rel_error[i]

    # Add residuals to the evaluation dictionary
    eval_dict['residuals'] = all_residuals

    return eval_dict

def test(model, data_loader, outfile, loss_fn, features, device, plot_dir='plots'):
    results = evaluate(model, data_loader, loss_fn, features, device)
    print("Test results:")
    for key,value in results.items():
        if key != 'residuals':
            print(key, f': {value:.4f}')
    if outfile:
        save_results_to_csv(outfile, results)
    if 'residuals' in results.keys():
        out_dir = os.path.dirname(outfile)
        residual_file = os.path.join(out_dir, 'residuals.csv')
        save_results_to_csv(residual_file, results['residuals'])

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return results

def test_current_and_best_model(model,test_loader,out_dir,loss_fn,features, device):
    logging.info("Testing latest model...")
    last_results = test(model,test_loader,os.path.join(out_dir, 'last_results.csv'), loss_fn,features, device)
    logging.info("Restoring model from previous checkpoint...")
    checkpoint = torch.load(os.path.join(out_dir, 'best_model.pkl'),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Testing best model...")
    best_results = test(model,test_loader,os.path.join(out_dir, 'best_results.csv'), loss_fn, features, device)
    logging.info("Done...\n")

    return last_results,best_results

if __name__ == '__main__':

    random_state = 4
    torch.manual_seed(random_state)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "gnn", "cnn-linear", "unet"], default='unet', help="Model type")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.model in ['cnn', 'cnn-linear']:
        config_file = 'config/train_triangle_regression.yaml'
    elif args.model == 'unet':
        config_file = 'config/train_triangle_regression_unet.yaml'
    else:
        config_file = 'config/train_triangle_regression_graph.yaml'
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device('cpu')
    print('-------%s-------' % config['model_name'] + '\n')
    logging.info('Loading Dataset and initializing model...')
    batch_size = 1 if config['data'].get('bucketize', False) else config['data']['batch_size']
    data_dir = 'data/TriangleFeatures'

    if args.model == 'gnn':
        train_set = TriangleFeatureGraphDataset(data_dir = data_dir, split = 'train', device = device, **config['data'])
        val_set = TriangleFeatureGraphDataset(data_dir = data_dir, split = 'valid', device = device, **config['data'])
        test_set = TriangleFeatureGraphDataset(data_dir = data_dir, split = 'test', device = device, **config['data'])
        train_loader = GraphLoader(train_set, batch_size, shuffle=True)
        val_loader = GraphLoader(val_set,batch_size=1, shuffle=False)
        test_loader = GraphLoader(test_set,batch_size=1, shuffle=False)
        in_channels = int(train_set.get_data_feature_dim())
        out_channels = int(train_set.get_data_output_dim())
        model = GraphTriangleEncoder(in_channels,out_channels,**config['model']).to(device)
    else:
        train_set = TriangleFeatureDataset(data_dir=data_dir, split='train', device = device, **config['data'])
        val_set = TriangleFeatureDataset(data_dir=data_dir, split='valid', device = device, **config['data'])
        test_set = TriangleFeatureDataset(data_dir=data_dir, split='test', device = device, **config['data'])
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        in_channels = int(train_set.get_data_feature_dim())
        out_channels = int(train_set.get_data_output_dim())
        if args.model == 'cnn':
            model = TriangleFeatureExtractor(in_channels,out_channels,**config['model']).to(device)
        elif args.model == 'unet':
            model = UNetTriangleModelRegression(**config['model']).to(device)
        else:
           raise NotImplementedError('Model type needs to be either cnn or unet')
    logging.info('Done...\n')
    logging.info('Initializing...')
    optimizer, scheduler, out_dir, log_dir, base_dir = initialize(model, config['training'], config['tag'],
                                                                            model_name=config['model_name'])
    loss_fn = get_loss_fn(config['loss_fn'])
    logging.info(('Done... \n'))

    setup_directories(log_dir, out_dir)
    if not os.path.isfile(os.path.join(base_dir, 'config.yaml')):
        save_config(base_dir, config)

    if config.get('resume_training', False) :
        checkpoint = torch.load(os.path.join(out_dir, config['chckpnt_file']))
        logging.info('Loading from checkpoint...')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info('Done... \n')
    else:
        start_epoch = 0

    features = train_set.get_feature_names()
    train(model, train_loader, val_loader, optimizer, loss_fn, config['training']['num_epochs'],
          log_dir, out_dir, device, scheduler, start_epoch,features)

    last_results, best_results = test_current_and_best_model(model, test_loader, out_dir,loss_fn, features, device)