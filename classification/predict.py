import torch
import yaml
import os
from .models import BuildingClassificationModel
from data.datasets import ClassificationDataset
from torch.utils.data import DataLoader
from collections import defaultdict
from src.plotting import plot_polygons_with_labels
import argparse

if __name__ =="__main__":
    torch.manual_seed(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", help="Path to config file", default='config/train_classification.yaml')
    parser.add_argument("-chkpt_file", help="filename of checkpoint to load", default='trained_models/classification.pkl')
    parser.add_argument("-outfile", help="outfile for plotting", default='plots/classification_results.svg')
    args = parser.parse_args()

    data_dir = 'data/classification'
    print("Classifying Buildings...")
    checkpoint_file = args.chkpt_file
    outfile = args.outfile
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
    checkpoint = torch.load(checkpoint_file,map_location='cpu')
    model = BuildingClassificationModel(**config['model']).to('cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_set = ClassificationDataset(data_dir =data_dir, split='test' ,**config['data'])
    data_loader = DataLoader(test_set, shuffle = True, batch_size = 1)
    num_classes = 10
    samples_per_class = 1
    collected_samples = defaultdict(list)

    # Loop until all classes have the required number of samples
    while any(len(collected_samples[c]) < samples_per_class for c in range(num_classes)):
        for data, label in data_loader:
            label = label.item()  # Assuming batch size is 1

            # Collect the sample if the class has not yet reached the target
            if len(collected_samples[label]) < samples_per_class:
                collected_samples[label].append(data.squeeze(0))

            # Stop if all classes have enough samples
            if all(len(collected_samples[c]) >= samples_per_class for c in range(num_classes)):
                break

    preds = defaultdict(list)
    with torch.no_grad():
        for label, samples in collected_samples.items():
            for sample in samples:
                pred = model(sample.unsqueeze(0).to('cpu')).argmax().item()
                preds[label].append(pred)
    for key,value in collected_samples.items():
        collected_samples[key] = [value[i].T for i in range(len(value))]
    plot_polygons_with_labels(collected_samples, preds, save_path=outfile, n_classes_per_row=5, show_labels = False)
