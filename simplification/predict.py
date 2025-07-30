import os
import yaml
import torch
import argparse
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score
from data.datasets import BuildingSimplificationDataset
from .train import test
from src.plotting import plot_iou_hd_quantiles, plot_footprints_with_node_labels, presentation_simplification_plot
from .models import BuildingSimplificationGraphModel,BuildingSimplificationModel
from .utils import get_loss_fn


def reconstruct_polygon(polygon, labels, pred_preMove, pred_nextMove):
    new_polygon = []
    for i in range(len(polygon)):
        current_vertex = polygon[i]
        label = labels[i]
        if label == 0:  # Remove
            continue
        elif label == 1:  # Keep
            new_polygon.append(current_vertex)
        elif label == 2:  # Move
            prev_idx = (i - 1) % len(polygon)
            next_idx = (i + 1) % len(polygon)
            prevs_vertex = polygon[prev_idx]
            next_vertex = polygon[next_idx]
            vec_pre = Point(current_vertex.x - prevs_vertex.x, current_vertex.y - prevs_vertex.y)
            vec_next = Point(next_vertex.x - current_vertex.x, next_vertex.y - current_vertex.y)
            vec_pre_mod = LineString([prevs_vertex, current_vertex]).length
            vec_next_mod = LineString([next_vertex, current_vertex]).length

            if vec_pre_mod > 0:
                pre_dir = Point(vec_pre.x/vec_pre_mod, vec_pre.y/vec_pre_mod)
            if vec_next_mod > 0:
                next_dir = Point(vec_next.x/vec_next_mod,vec_next.y/vec_next_mod)

            nextMove = Point(pred_nextMove[i] * next_dir.x,pred_nextMove[i] * next_dir.y)
            preMove = Point(pred_preMove[i] * pre_dir.x,pred_preMove[i] * pre_dir.y)

            new_polygon.append(Point(current_vertex.x + nextMove.x+preMove.x,current_vertex.y + nextMove.y+preMove.y))

    new_polygon.append(new_polygon[0])

    return new_polygon

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Predict simplifications on test set')
    parser.add_argument('-t', action='store_true', default=True, help='run test method.')
    parser.add_argument('-p', action='store_true', default = True, help='predict test set.')
    parser.add_argument('--chkpnt_path', type=str, default='trained_models/simplification_pretrained.pkl', help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default='config/train_simplification_pretrained.yaml', help='Path to config file')
    parser.add_argument('--plot_dir', type=str,default='plots/', help='Out dir for plots')
    parser.add_argument('--model', type=str, choices=['pretrained','cnn','gnn'],
                        default='pretrained', help='Model type to use for prediction. If "gnn" or "cnn" is used, correct config and checkpoints need to be provided')
    parser.add_argument('--outfile', type=str, default='simplification_results.shp',
                        help='Output file to save predictions (default: predictions.shp)')
    parser.add_argument('--outdir', type=str, default='results/BuildingSimplification')
    args = parser.parse_args()

    random_state = 4
    torch.manual_seed(random_state)
    assert args.p or args.t, "usage: Must specify at least one of -t (test) or -p (predict)"

    if args.model == 'pretrained':
        args.config_path = 'config/train_simplification_pretrained.yaml'
        args.chkpnt_path = 'trained_models/simplification.pkl'


    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')

    # Load model
    if args.model == 'gnn':
        model = BuildingSimplificationGraphModel(**config['model']).to(device)
    elif args.model in ['pretrained','cnn','features', 'unet_vanilla']:
        model = BuildingSimplificationModel(**config['model']).to(device)
    else:
        raise NotImplementedError('Model type not supported')

    # Load checkpoint
    logging.info(f"Loading checkpoint from {args.chkpnt_path}")
    chkpt = torch.load(args.chkpnt_path, map_location=device)
    model.load_state_dict(chkpt['model_state_dict'])
    model.eval()

    # Load dataset
    data_dir = 'data/BuildingSimplification'
    test_set = BuildingSimplificationDataset(data_dir, split='test', **config['data'])
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False
    )

    if args.t:
        test_out = os.path.join(args.outdir, 'test_results.csv')
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir, exist_ok=True)
        loss_fn_move = get_loss_fn(config['training']['loss_fn_reg'])
        loss_fn_remove = get_loss_fn(config['training']['loss_fn_cls'])
        results = test(model,test_loader,test_out,device,loss_fn_move,loss_fn_remove, average='macro')
    logging.info(f"Loaded test dataset with {len(test_set)} samples")

    # Store prediction
    polygon_len_list = []
    footprint_pred_list = []
    polygon_pred_list = []
    footprint_gt_list = []
    polygon_gt_list = []
    f1_scores = []
    preMove_mae_list = []
    nextMove_mae_list = []
    footprints = []
    footprint_polygons = []
    y_preds = []
    y_gts = []
    osmid_list = []
    class_prediction_counts = {0: 0, 1: 0, 2: 0}
    gt_class_counts = {0: 0, 1: 0, 2: 0}

    #prediction loop
    if args.p:
        logging.info("Running predictions...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                x, y, metadata = batch['x'].to(device), batch['y'].to(device), batch['meta_data']
                additional_inputs = []

                if x.ndim > 3:
                    x = x.squeeze(0)
                    y = y.squeeze(0)
                if x.ndim < 3:
                    x = x.unsqueeze(0)
                    y = y.unsqueeze(0)

                if batch.get('edge_index', None) is not None:
                    edge_index = batch['edge_index']
                    additional_inputs.append(edge_index)
                if batch.get('edge_attr', None) is not None:
                    edge_attr = batch['edge_attr']
                    additional_inputs.append(edge_attr)

                pred_rm, pred_preMove, pred_nextMove = model(x, *additional_inputs)


                pred_labels = pred_rm.argmax(1).view(-1).numpy()
                rm_targets = y[..., 0].long().view(-1).numpy()
                f1_scores.append(f1_score(rm_targets, pred_labels,average='macro'))

                for label in rm_targets:
                    gt_class_counts[label] += 1

                for pred in pred_labels:
                    class_prediction_counts[pred] += 1

                preMove_mae_list.append(torch.mean(torch.abs(pred_preMove - y[..., 1])).item())
                nextMove_mae_list.append(torch.mean(torch.abs(pred_nextMove - y[..., 2])).item())

                #swap channels from convolutional to sequential
                x,y,metadata = x.squeeze().permute(1,0), y.squeeze(),metadata.squeeze()
                pred_preMove, pred_nextMove = pred_preMove.squeeze(), pred_nextMove.squeeze()
                y_preds.append(pred_labels)
                y_gts.append(y.numpy())
                geometry = [Point(metadata[vid][2], metadata[vid][3]) for vid in range(len(metadata))]

                polygon_gt = reconstruct_polygon(geometry, rm_targets, y[:,1].numpy(), y[:,2].numpy())
                polygon_pred = reconstruct_polygon(geometry, pred_labels, pred_preMove.numpy(), pred_nextMove.numpy())
                polygon_pred_list.append(Polygon(polygon_pred))
                polygon_gt_list.append(Polygon(polygon_gt))
                footprint_pred_list.append(polygon_pred)
                footprint_gt_list.append(polygon_gt)
                polygon_len_list.append(len(x))
                geometry.append(geometry[0])
                footprints.append(geometry)
                footprint_polygons.append(Polygon(geometry))
                osmid_list.append(metadata[0,0].long().item())

        gpd_dict = {'osmid': osmid_list,
                         'geometry': footprint_polygons,
                         'reconstructed_pred': polygon_pred_list,
                         'reconstructed_gt': polygon_gt_list,
                         'y_pred': y_preds,
                         'f1_score': f1_scores,
                         'mae_nextMove': nextMove_mae_list,
                         'mae_preMove': preMove_mae_list}
        results = gpd.GeoDataFrame(gpd_dict)
        jaccard_coeff_list = []
        hd_dist_list = []
        for i,row in results.iterrows():
            try:
                intersection = row.loc['reconstructed_pred'].intersection(row.loc['reconstructed_gt'])
                union = row.loc['reconstructed_pred'].union(row.loc['reconstructed_gt'])
            except:
                print('invalid topology: ', row.loc['osmid'])
                print(row.loc['geometry'])
                jaccard_coeff_list.append(0)
                hd_dist_list.append(100)
                continue
            jaccard_coeff_list.append(intersection.area / union.area)
            hd_dist_list.append(row.loc['reconstructed_pred'].hausdorff_distance(row.loc['reconstructed_gt']))

        results['jaccard_coeff'] = jaccard_coeff_list
        results['hd_dist'] = hd_dist_list

        hd_sorted = results.sort_values(by=['hd_dist'])
        jaccard_sorted = results.sort_values(by=['jaccard_coeff'], ascending = False)
        jaccard_sorted.jaccard_coeff = 1-jaccard_sorted.jaccard_coeff
        plot_iou_hd_quantiles(hd_sorted, jaccard_sorted, args.plot_dir, quantiles = [0.25,0.5,0.75,1.], plot_uppers=False)
        plot_footprints_with_node_labels(footprints[10], y_gts[10][:,0], y_preds[10],save_path = os.path.join(args.plot_dir, 'pred_gt_node_labels.png'))
        presentation_simplification_plot(footprints[10],y_gts[10],results.loc[10,'reconstructed_gt'], show_labels=True, save_path=os.path.join(args.plot_dir, 'presentation_simplification.png'))
        print('Fraction of footprints with perfect reconstruction: ', np.sum(results.jaccard_coeff==1)/len(results))
        print(f'IoU: {results.jaccard_coeff.mean()} +- {results.jaccard_coeff.std()}')
        print(f'HD: {results.hd_dist.mean()} +- {results.hd_dist.std()}')
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        results.to_file(os.path.join(args.outdir, args.outfile))

if __name__ == "__main__":
    main()

