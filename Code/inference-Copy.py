import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import MVTecDataset_test, get_data_transforms
import numpy as np
from scipy.stats import pearsonr

def lin_concordance_correlation_coefficient(x, y):
    # Calculate Pearson correlation coefficient
    rho, _ = pearsonr(x, y)
    
    # Calculate means
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    # Calculate standard deviations
    sigma_x = np.std(x, ddof=1)
    sigma_y = np.std(y, ddof=1)
    
    # Calculate CCC
    numerator = 2 * rho * sigma_x * sigma_y
    denominator = sigma_x**2 + sigma_y**2 + (mu_x - mu_y)**2
    ccc = numerator / denominator

    return round(ccc, 4)  # Rounding to 4 decimal places


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default='./your_checkpoint_folder', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--classes', nargs="+", default=["carpet", "leather"])
    parser.add_argument('--color_space', default='RGB', type= str)
    pars = parser.parse_args()
    return pars

def create_checkpoint_folder(checkpoint_folder, _class_):
    class_checkpoint_folder = os.path.join(checkpoint_folder, _class_)
    os.makedirs(class_checkpoint_folder, exist_ok=True)
    return class_checkpoint_folder

def load_model_weights(model, checkpoint_path):
    ckp = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckp)
    return model

def read_data(file_path):
    """
    Reads the data from a text file, skipping the header row, and returns tuples of (estimation, ground_truth) for three pairs.
    """
    data = np.loadtxt(file_path, skiprows=1)
    pairs = [(data[:, i], data[:, i+1]) for i in range(1, 7, 2)]
    return pairs

def mean_absolute_error(true_values, estimations):
    return np.mean(np.abs(true_values - estimations))

def mean_squared_error(true_values, estimations):
    return np.mean((true_values - estimations) ** 2)

def root_mean_squared_error(true_values, estimations):
    return np.sqrt(mean_squared_error(true_values, estimations))

def mean_absolute_percentage_error(true_values, estimations):
    non_zero_mask = true_values != 0  # Create a mask to filter out zero values
    return np.mean(np.abs((true_values[non_zero_mask] - estimations[non_zero_mask]) / true_values[non_zero_mask])) * 100

def r_squared(true_values, estimations):
    ss_res = np.sum((true_values - estimations) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    return 1 - (ss_res / ss_tot)

def plot_residuals(estimations, true_values, data_type, file_name):
    residuals = estimations - true_values
    plt.figure()
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.xlabel('Data Point Index (n)')
    plt.ylabel(f'Residual (Estimation ({data_type}) - True Value ({data_type}))')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.savefig(file_name)
    plt.show()

def inference(_class_, pars):
    #checkpoint_folder = create_checkpoint_folder(pars.checkpoint_folder, _class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform, gt_leaf_transform = get_data_transforms(pars.image_size, pars.image_size)
    test_path = "../../../../Data/" + _class_
    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'

    test_data = MVTecDataset_test(root=test_path, transform=data_transform,  gt_transform=gt_transform, gt_leaf_transform = gt_leaf_transform,  color_space = pars.color_space)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    print("Loaded pretrained model.")

      # Use pretrained wide_resnet50 for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)

    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    # Load trained weights for projection layer, bn (OCBE), decoder (student)    
    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])

    area_leafs, area_leafs_gt, area_anomaly, area_anomaly_gt, auroc_px, auroc_sp, aupro_px,f1, precision = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device, _class_, pars.color_space)       
    
    return area_leafs, area_leafs_gt, area_anomaly, area_anomaly_gt, auroc_px, auroc_sp, aupro_px,f1, precision

if __name__ == '__main__':
    pars = get_args()
    #setup_seed(111)
    metrics = {'class': [], 'AUROC_sample': [], 'AUROC_pixel': [], 'AUPRO_pixel': [], "F1": [], "precision": []}

    for c in pars.classes:
        area_leafs, area_leafs_gt, area_anomaly, area_anomaly_gt, auroc_sp, auroc_px, aupro_px,f1, precision = inference(c, pars)
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        metrics['F1'].append(f1)
        metrics['precision'].append(precision)
        metrics_df = pd.DataFrame(metrics)
        
        metrics_df.to_csv(os.path.join(pars.checkpoint_folder, 'metrics_checkpoints.csv'), index=False)

        result_df = pd.DataFrame({
            'n': list(area_leafs.keys()),  # 'n' column containing the keys from area_leafs
            'area_leafs': list(area_leafs.values()),
            'area_leafs_gt': list(area_leafs_gt.values()),
            'area_anomaly': list(area_anomaly.values()),
            'area_anomaly_gt': list(area_anomaly_gt.values()),
        })
        result_df.reset_index(drop=True, inplace=True)

        result_df['percentage'] = result_df['area_anomaly'] / result_df['area_leafs'] * 100
        result_df['percentage_gt'] = result_df['area_anomaly_gt'] / result_df['area_leafs_gt'] * 100
        # Sample dictionaries

        print(result_df)
  
        with open("result_df.txt", "w") as f:
            f.write(result_df.to_string(index=False))

        with open("anomaly_area.txt", "w") as f:
            for filename, area in area_anomaly.items():
                f.write(f"File: {filename}, Area: {area}\n")

        with open("leaf_area.txt", "w") as f:
            for filename, area in area_leafs.items():
                f.write(f"File: {filename}, Area: {area}\n")
        file_path = 'result_df.txt'
        pairs = read_data(file_path)
        pair_names = ["Leaf Area", "Disease Area", "Diseased Leaf Area Percentage"]  # Customize these names as needed
        data_type = ["px", "px", "%"]
        for pair_name, data_type,(estimations, true_values) in zip(pair_names,data_type, pairs):
            mae = mean_absolute_error(true_values, estimations)
            mse = mean_squared_error(true_values, estimations)
            rmse = root_mean_squared_error(true_values, estimations)
            mape = mean_absolute_percentage_error(true_values, estimations)
            r2 = r_squared(true_values, estimations)
            ccc_value = lin_concordance_correlation_coefficient(true_values, estimations)

            print(f"Pair {pair_name}:")
            print("  MAE:", mae)
            print("  MSE:", mse)
            print("  RMSE:", rmse)
            print("  MAPE:", mape)
            print("  R-squared:", r2)
            print("Lin's Concordance Correlation Coefficient:", ccc_value)

            print()

            file_name = f"{pair_name}_residuals_plot.png"  # Customize file name as needed
            plot_residuals(estimations, true_values, data_type, file_name)
