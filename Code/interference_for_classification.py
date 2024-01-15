import os
import torch
import numpy as np
import random
import pandas as pd
from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from utils.utils_test_classification import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer
from dataset.dataset_classification import MVTecDataset_test, get_data_transforms

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

def inference(_class_, pars):
    #checkpoint_folder = create_checkpoint_folder(pars.checkpoint_folder, _class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, _, _ = get_data_transforms(pars.image_size, pars.image_size)
    test_path = "../../../../Data/" + _class_
    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'

    test_data = MVTecDataset_test(root=test_path, transform=data_transform,   color_space = pars.color_space)
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

    area_leafs,area_anomaly, f1, precision = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device, _class_, pars.color_space)       
    
    return area_leafs,area_anomaly, f1, precision

if __name__ == '__main__':
    pars = get_args()
    #setup_seed(111)
    metrics = {'class': [],"F1": [], "precision": []}

    for c in pars.classes:
      area_leafs,area_anomaly, f1, precision= inference(c, pars)
      metrics['class'].append(c)
      metrics['F1'].append(f1)
      metrics['precision'].append(precision)
      metrics_df = pd.DataFrame(metrics)
      metrics_df.to_csv(os.path.join(pars.checkpoint_folder, 'metrics_checkpoints.csv'), index=False)

      result_df = pd.DataFrame({
          'area_leafs': area_leafs.values(),
          'area_anomaly': area_anomaly.values(),
      }, index=area_leafs.keys())
      
      result_df['percentage'] = result_df['area_anomaly'] / result_df['area_leafs'] * 100

      print(result_df)
      
      with open("result_df.txt", "w") as f:
          f.write(result_df.to_string())

      with open("anomaly_area.txt", "w") as f:
          for filename, area in area_anomaly.items():
              f.write(f"File: {filename}, Area: {area}\n")

      with open("leaf_area.txt", "w") as f:
          for filename, area in area_leafs.items():
              f.write(f"File: {filename}, Area: {area}\n")
