from data.datasets import ClassificationDataset
from src.utils import save_results_to_csv, save_config, save_checkpoint, setup_directories,initialize
import os
import yaml
import torch
import numpy as np
import pandas as pd
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from .models import BuildingClassificationModel


def train(model,
          train_set,
          val_set,
          optimizer,
          loss_fn,
          num_epochs,
          batch_size,
          log_dir,
          out_dir,
          device,
          scheduler = None,
          checkpoint=None,
          resume_training = False):

    best_results = {'acc': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    best_model_file = os.path.join(out_dir, 'best_model.pkl')
    latest_model_file = os.path.join(out_dir, 'latest_model.pkl')

    if resume_training:
        logging.info('Loading from checkpoint...')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info('Done... \n')
    else:
        start_epoch = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    logging.info("Start training...")
    logger = SummaryWriter(log_dir)
    prog_bar = tqdm(initial=start_epoch, total=start_epoch + num_epochs, desc="Training")
    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_total_loss = []

        # Training step
        for x,y in train_loader:
            if x.ndim > 3:
                x = x.squeeze()
            if y.ndim > 1:
                y = y.squeeze(dim=1)
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_total_loss.append(loss)

        mean_epoch_loss = torch.mean(torch.tensor(epoch_total_loss))

        # Validation step
        eval_dict = evaluate(model, val_loader, device)

        for key, value in eval_dict.items():
            if value > best_results[key]:
                best_results[key] = value
                #only best model on f1 score is stored
                if key == 'f1_score':
                    save_checkpoint(epoch, model, optimizer, best_model_file, scheduler)

        # Log training and validation metrics
        logger.add_scalar('Train/loss', mean_epoch_loss, epoch)
        if scheduler:
            logger.add_scalar('Train/lr', scheduler.get_last_lr()[0], epoch)
        for key, value in eval_dict.items():
            logger.add_scalar(f'Val/{key}', value, epoch)

        # update scheduler at the end of every epoch
        if scheduler:
            scheduler.step()

        # Update progress bar
        prog_bar.set_description(f"Train Loss: {mean_epoch_loss:.4f}; Val acc: {eval_dict['acc']:.4f}")
        prog_bar.update()

    # Save the latest model checkpoint
    save_checkpoint(epoch, model, optimizer, latest_model_file, scheduler)
    logger.close()
    logging.info("Done...\n")

def evaluate(model, data_loader: torch.utils.data.DataLoader, device) -> dict:

        model.eval()
        preds,targets  = [], []
        correct_cnt = 0
        total_cnt = 0
        with torch.no_grad():
            for x, y in data_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x)
                pred_choice = pred.max(1)[1]
                y = y.long()
                correct_cnt += pred_choice.eq(y.view(-1)).sum().item()
                total_cnt += y.size(0)
                pred = torch.max(pred, 1)[1].view(-1)
                preds += pred.detach().cpu().numpy().tolist()
                targets += y.cpu().numpy().tolist()

        acc = correct_cnt / total_cnt
        precision = precision_score(targets, preds, average='macro')
        recall = recall_score(targets, preds, average='macro')
        f1 = f1_score(targets, preds, average='macro')
        metrics = {'acc': acc, 'precision': precision, 'recall': recall, 'f1_score': f1}
        return metrics

def test(model, test_set, device, outfile=None):
    data_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    results = evaluate(model, data_loader, device)
    print("Test results:")
    for key,value in results.items():
        print(key, f': {value:.4f}')
    if outfile:
        save_results_to_csv(outfile, results)

    return results

def test_current_and_best_model(model,test_set,device,out_dir):
    logging.info("Testing latest model...")
    last_results = test(model,test_set,device, outfile=os.path.join(out_dir, 'last_results.csv'))
    logging.info("Restoring model from previous checkpoint...")
    checkpoint = torch.load(os.path.join(out_dir, 'best_model.pkl'),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Testing best model...")
    best_results = test(model,test_set,device, outfile=os.path.join(out_dir, 'best_results.csv'))
    logging.info("Done...\n")

    return last_results,best_results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=['vanilla','appendcoords'], default='vanilla', help="Model type")
    args = parser.parse_args()
    
    if args.model_type == 'vanilla':
        config_file = 'config/train_classification.yaml'
    elif args.model_type == 'appendcoords':
        config_file = 'config/train_classification_append.yaml'
    else:
        raise NotImplementedError(f'Valid model types are "vanilla" and "appendcoords", not {args.model_type}')
                                  
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    print('-------%s-------'%config['model_name'] + '\n')
    logging.info('Loading Dataset...')
    train_set = ClassificationDataset(data_dir = 'resources/DPCN/data/', split='train', **config['data'])
    val_set = ClassificationDataset(data_dir = 'resources/DPCN/data/', split='test', **config['data'])
    #test_set = Model10DataSet(data_dir = 'resources/DPCN/data/', split='test', **config['data'])
    test_set = val_set
    logging.info('Done...\n')
    batch_size = 1 if config['data'].get('raw_coordinates', False) else config['batch_size']
    random_state = 4
    torch.manual_seed(random_state)
    n_iter = 5
    logging.info('Initializing...')
    device = torch.device(config['device']) if torch.cuda.is_available() else 'cpu'

    total_results = {'acc': [], 'precision': [], 'recall': [], 'f1_score': []}
    mean = {'acc': 0., 'precision': 0., 'recall': 0., 'f1_score': 0.}
    std = {'acc': 0., 'precision': 0., 'recall': 0., 'f1_score': 0.}

    total_results_best = {'acc': [], 'precision': [], 'recall': [], 'f1_score': []}
    mean_best = {'acc': 0., 'precision': 0., 'recall': 0., 'f1_score': 0.}
    std_best = {'acc': 0., 'precision': 0., 'recall': 0., 'f1_score': 0.}
    for i in range(n_iter):
        logging.info(f'Iteration {i+1}/{n_iter}')

        if batch_size == 1 and config['model']['norm'] == 'batch':
            logging.info('Batchnorm was selected, but batch size is 1. Switching to group normalization...')
            config['model']['norm'] = 'group'

        if args.model_type == 'vanilla':
            model = BuildingClassificationModel(**config['model']).to(device)
        elif args.model_type == 'appendcoords':
            model = ClassificationAppendCoordsEveryLayerModel(**config['model']).to(device)
        else:
            raise NotImplementedError('Model type not implemented')

        optimizer, scheduler, out_dir, log_dir,base_dir = initialize(model, config['training'], config['tag'], model_name = config['model_name'] , run = i)
        loss_fn = CrossEntropyLoss()

        setup_directories(log_dir, out_dir)
        if not os.path.isfile(os.path.join(base_dir, 'config.yaml')):
            save_config(base_dir, config)
        train(model,train_set,val_set, optimizer,loss_fn,config['training']['num_epochs'],batch_size,
              log_dir,out_dir,device,scheduler)

        last_results,best_results = test_current_and_best_model(model,test_set,device,out_dir)
        for key,value in last_results.items():
            total_results[key].append(value)
        for key,value in best_results.items():
            total_results_best[key].append(value)

    print('----------Overall Results-------------')
    for key,value in total_results.items():
        mean[key] = np.mean(value)
        std[key] = np.std(value)

    for key,value in total_results_best.items():
        mean_best[key] = np.mean(value)
        std_best[key] = np.std(value)

    overall_results = pd.DataFrame({'mean': mean, 'std': std}).T
    overall_results_best = pd.DataFrame({'mean': mean_best, 'std': std_best}).T
    print(overall_results)
    print('Best Results:')
    print(overall_results_best)
    outfile = os.path.join(base_dir, 'n=5_results.csv')
    overall_results.to_csv(outfile)
    outfile = os.path.join(base_dir, 'n=5_results_best.csv')
    overall_results_best.to_csv(outfile)
    logging.info('Done...\n')
    #test_current_and_best_model(model, test_set, device, out_dir)




