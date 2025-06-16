import argparse
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.utils import save_checkpoint,initialize,setup_directories, save_config
from data.datasets import get_dataloaders
from .models import BuildingSimplificationModel, BuildingSimplificationGraphModel
from .utils import automatic_weight,get_loss_fn,print_results,initialize_weights,convert_to_native,ClassWeightedMAELoss
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
import logging
import torch
import os

def train(model, train_loader, val_loader, optimizer, loss_fn_move, loss_fn_remove, num_epochs,
          log_dir, out_dir, device, scheduler, start_epoch = 0):
    writer = SummaryWriter(log_dir=log_dir)

    val_loss = 0
    val_loss_regression = np.inf
    logging.info('Start training...')
    prog_bar = tqdm(initial=start_epoch, total=start_epoch + num_epochs, desc="Training")
    for epoch in range(start_epoch, start_epoch+num_epochs):

        model.train()
        epoch_total_loss = []
        for batch in train_loader:

            x, y = batch['x'].to(device), batch['y'].to(device)
            additional_inputs = []
            if x.ndim > 3:
                x = x.squeeze(0)
                y = y.squeeze(0)
            if x.ndim < 3 and isinstance(model,BuildingSimplificationModel):
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
            if batch.get('edge_index', None) is not None:
                edge_index = batch['edge_index'].to(device)
                additional_inputs.append(edge_index)
            if batch.get('edge_attr', None) is not None:
                edge_attr = batch['edge_attr'].to(device)
                additional_inputs.append(edge_attr)
            pred_rm, pred_preMove, pred_nextMove = model(x, *additional_inputs)
            rm_loss = loss_fn_remove(pred_rm, y[...,0].long())
            if isinstance(loss_fn_move, ClassWeightedMAELoss):
                preMove_loss = loss_fn_move(pred_preMove, y[..., 1], y[...,0])
                nextMove_loss = loss_fn_move(pred_nextMove, y[..., 2], y[...,0])
            else:
                preMove_loss = loss_fn_move(pred_preMove, y[...,1])
                nextMove_loss = loss_fn_move(pred_nextMove, y[...,2])

            task_loss = torch.stack((rm_loss, preMove_loss, nextMove_loss))
            total_loss = automatic_weight(model.loss_weights, task_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total_loss.append(total_loss.item())

        epoch_avg_loss = sum(epoch_total_loss) / len(epoch_total_loss)

        # Validation step
        eval_loss, eval_removal_dict, eval_preMove_dict, eval_nextMove_dict = evaluate(model, loss_fn_move, loss_fn_remove, val_loader,device)

        # Log results to TensorBoard
        writer.add_scalar('Train/loss', epoch_avg_loss, epoch)
        writer.add_scalar('Val/loss', eval_loss, epoch)
        if scheduler:
            writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], epoch)
            scheduler.step()
        for metric in eval_removal_dict:
            writer.add_scalar(f'Val/{metric}', eval_removal_dict[metric], epoch)
        for metric in eval_preMove_dict:
            writer.add_scalar(f'Val/preMove_{metric}', eval_preMove_dict[metric], epoch)
        for metric in eval_nextMove_dict:
            writer.add_scalar(f'Val/nextMove_{metric}', eval_nextMove_dict[metric], epoch)

        # Save the best model
        if eval_removal_dict['f1'] > val_loss:
            val_loss = eval_removal_dict['f1']
            best_model_path = f"{out_dir}/best_model.pkl"
            save_checkpoint(epoch, model, optimizer, best_model_path, scheduler)

        combined_regression_score = 0.5*(eval_nextMove_dict['mae']+eval_preMove_dict['mae'])
        if combined_regression_score < val_loss_regression:
            val_loss_regression = combined_regression_score
            best_model_path = f"{out_dir}/best_model_regression.pkl"
            save_checkpoint(epoch, model, optimizer, best_model_path, scheduler)

        prog_bar.set_description(f"Train Loss: {epoch_avg_loss:.4f}; Val acc: {eval_removal_dict['accuracy']:.4f}")
        prog_bar.update()

    # Save the last model
    last_model_path = f"{out_dir}/last_model.pkl"
    save_checkpoint(epoch, model, optimizer, last_model_path, scheduler)

    writer.close()
    prog_bar.close()
    logging.info('Done...')

def evaluate(model, loss_fn_move, loss_fn_remove, data_loader, device, average= 'macro'):
    model.eval()

    preMove_mse = []
    nextMove_mse = []
    preMove_mae = []
    nextMove_mae = []
    preMove_mse_filter = []
    nextMove_mse_filter = []
    preMove_mae_filtered = []
    nextMove_mae_filtered = []
    total_losses = []
    pred_rm_list = []
    gt_rm_list = []

    eval_removal_metrics = {
        'accuracy': [],
        'f1': []
    }

    with torch.no_grad():
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
            pred_rm, pred_preMove, pred_nextMove = model(x,*additional_inputs)

            # Compute individual losses
            rm_loss = loss_fn_remove(pred_rm, y[...,0].long())
            pred_labels = pred_rm.argmax(1).view(-1).detach().cpu().numpy()

            if isinstance(loss_fn_move, ClassWeightedMAELoss):
                preMove_loss = loss_fn_move(pred_preMove, y[..., 1], y[..., 0])
                nextMove_loss = loss_fn_move(pred_nextMove, y[..., 2], y[..., 0])
            else:
                preMove_loss = loss_fn_move(pred_preMove, y[..., 1])
                nextMove_loss = loss_fn_move(pred_nextMove, y[..., 2])

            # Evaluate removal metrics
            rm_targets = y[..., 0].long().view(-1).detach().cpu().numpy()
            eval_removal_metrics['accuracy'].append(accuracy_score(rm_targets, pred_labels))
            eval_removal_metrics['f1'].append(f1_score(rm_targets, pred_labels, average = average))
            pred_rm_list.append(pred_labels)
            gt_rm_list.append(rm_targets)

            # Collect losses
            preMove_mse.append(torch.mean((pred_preMove-y[...,1])**2).detach().cpu().numpy())
            nextMove_mse.append(torch.mean((pred_nextMove-y[...,2])**2).detach().cpu().numpy())
            preMove_mae.append(torch.mean(torch.abs(pred_preMove-y[...,1])).detach().cpu().numpy())
            nextMove_mae.append(torch.mean(torch.abs(pred_nextMove-y[...,2])).detach().cpu().numpy())
            total_loss = rm_loss + preMove_loss + nextMove_loss
            total_losses.append(total_loss.item())

            dis_filter = (torch.abs(y[..., 1]) + torch.abs(y[..., 2])) >= 0.01
            move_filter = y[...,0] == 2
            if dis_filter.any():
                preMove_filtered_loss = torch.mean((pred_preMove[dis_filter]-y[..., 1][dis_filter])**2)
                nextMove_filtered_loss = torch.mean((pred_nextMove[dis_filter]-y[..., 2][dis_filter])**2)
                preMove_mse_filter.append(preMove_filtered_loss.item())
                nextMove_mse_filter.append(nextMove_filtered_loss.item())
            if move_filter.any():
                preMove_mae_filtered.append(
                    torch.mean(torch.abs(pred_preMove[move_filter] - y[move_filter][..., 1])).detach().cpu().numpy())
                nextMove_mae_filtered.append(
                    torch.mean(torch.abs(pred_nextMove[move_filter] - y[move_filter][..., 2])).detach().cpu().numpy())
    # Aggregate metrics
    eval_removal_dict = {
        'accuracy': np.mean(eval_removal_metrics['accuracy']),
        'f1': np.mean(eval_removal_metrics['f1']),
        'f1-micro': f1_score(np.concatenate(gt_rm_list), np.concatenate(pred_rm_list), average='micro'),
        'f1-macro': f1_score(np.concatenate(gt_rm_list), np.concatenate(pred_rm_list), average='macro'),
        'f1-weighted': f1_score(np.concatenate(gt_rm_list), np.concatenate(pred_rm_list), average='weighted'),
    }

    eval_preMove_dict = {
        'rmse': np.sqrt(np.mean(preMove_mse)),
        'mae': np.mean(preMove_mae),
        'mae_filtered': np.mean(preMove_mae_filtered)
    }

    eval_nextMove_dict = {
        'rmse': np.sqrt(np.mean(nextMove_mse)),
        'mae': np.mean(nextMove_mae),
        'mae_filtered': np.mean(nextMove_mae_filtered)
    }

    if preMove_mse_filter:
        eval_preMove_dict['rmse_filter'] = np.sqrt(np.mean(preMove_mse_filter))

    if nextMove_mse_filter:
        eval_nextMove_dict['rmse_filter'] = np.sqrt(np.mean(nextMove_mse_filter))

    eval_loss = np.mean(total_losses)

    return eval_loss, eval_removal_dict, eval_preMove_dict, eval_nextMove_dict

def test(model,data_loader,outfile,device,loss_fn_move,loss_fn_remove, average='macro'):
    loss, eval_removal_dict, eval_preMove_dict, eval_nextMove_dict = evaluate(model, loss_fn_move, loss_fn_remove, data_loader,device, average)
    results = {'remove': eval_removal_dict, 'preMove': eval_preMove_dict, 'nextMove': eval_nextMove_dict, 'loss': loss}
    # Convert the results dictionary
    results = convert_to_native(results)
    if outfile:
        with open(outfile, 'w') as f:
            yaml.dump(results,f)
    print_results(results)
    return results

def test_current_and_best_model(model,test_loader,out_dir,device,loss_fn_move,loss_fn_remove, average='macro'):
    logging.info("Testing latest model...")
    last_results = test(model, test_loader, os.path.join(out_dir, 'last_results.csv'), device, loss_fn_move,loss_fn_remove, average)
    logging.info("Restoring model from previous checkpoint...")
    checkpoint = torch.load(os.path.join(out_dir, 'best_model.pkl'),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Testing best model...")
    best_results = test(model, test_loader, os.path.join(out_dir, 'best_results.csv'), device, loss_fn_move,loss_fn_remove, average)
    logging.info("Done...\n")
    logging.info("Restoring model from previous checkpoint...")
    checkpoint = torch.load(os.path.join(out_dir, 'best_model_regression.pkl'),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Testing best regression model...")
    best_results_regression = test(model, test_loader, os.path.join(out_dir, 'best_results_regression.csv'), device, loss_fn_move,
                        loss_fn_remove, average)
    logging.info("Done...\n")

    return last_results,best_results, best_results_regression

if __name__ == '__main__':

    random_state = 4
    torch.manual_seed(random_state)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["features", "pretrained", 'gcn','unet_vanilla','sage_conv','spline_conv'],
                        default='pretrained', help="Model type")
    args = parser.parse_args()
    if args.model == 'features':
        config_file = 'config/train_simplification_features.yaml'
    elif args.model == 'pretrained':
        config_file = 'config/train_simplification_pretrained.yaml'
    elif args.model == 'unet_vanilla':
        config_file = 'config/train_simplification_vanilla.yaml'
    elif args.model == 'gcn':
        config_file = 'config/train_simplification_gcn.yaml'
    elif args.model == 'sage_conv':
        config_file = 'config/train_simplification_sageconv.yaml'
    elif args.model == 'spline_conv':
        config_file = 'config/train_simplification_splineconv.yaml'
    else:
        raise NotImplementedError('Model must be one of cnn, transformer, gnn')

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    print('-------%s-------' % config['model_name'] + '\n')
    logging.info('Loading Dataset...')
    batch_size = 1 if config['data'].get('bucketize', False) else config['data']['batch_size']
    train_loader,val_loader,test_loader = get_dataloaders(args.model, data_dir='data/BuildingSimplification', **config['data'])
    logging.info('Done...\n')
    logging.info('Initializing...')
    device = torch.device(config['device']) if torch.cuda.is_available() else 'cpu'
    if args.model in ['gcn','spline_conv', 'sage_conv']:
        model = BuildingSimplificationGraphModel(**config['model']).to(device)#
    else:
        model = BuildingSimplificationModel(**config['model']).to(device)

    optimizer, scheduler, out_dir, log_dir, base_dir = initialize(model, config['training'], config['tag'],
                                                                            model_name=config['model_name'])

    loss_fn_move = get_loss_fn(config['training']['loss_fn_reg'])
    loss_fn_remove = get_loss_fn(config['training']['loss_fn_cls'], device=device)
    if config.get('pretrained_weights_fname',False):
        model = initialize_weights(model, config['pretrained_weights_fname'], device)
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
    train(model, train_loader, val_loader, optimizer, loss_fn_move, loss_fn_remove, config['training']['num_epochs'],
          log_dir, out_dir, device, scheduler, start_epoch)

    #last_results, best_results = test_current_and_best_model(out_dir, device, model, loss_fn_move, loss_fn_remove, test_loader)
    last_results, best_results, best_results_regression = test_current_and_best_model(model,test_loader,out_dir,device,loss_fn_move,loss_fn_remove, 'macro')
