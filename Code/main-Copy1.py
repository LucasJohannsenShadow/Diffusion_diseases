import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
import geomloss
import cv2
from fastprogress import progress_bar
from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from dataset.dataset import MVTecDataset_test, MVTecDataset_train, get_data_transforms

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--save_folder', default = './RD++_checkpoint_result', type=str)
    parser.add_argument('--checkpoint_folder', default='./your_checkpoint_folder', type=str)
    parser.add_argument('--batch_size', default = 8, type=int)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--detail_training', default='note', type = str)
    parser.add_argument('--proj_lr', default = 0.001, type=float)
    parser.add_argument('--distill_lr', default = 0.01, type=float)
    parser.add_argument('--weight_proj', default = 0.2, type=float) 
    parser.add_argument('--classes', nargs="+", default=["carpet", "leather"])
    parser.add_argument('--use_pretrained', default= "False", type= str)
    parser.add_argument('--color_space', default='RGB', type= str)
    parser.add_argument('--noise', default= 'Simplex', type= str)
    pars = parser.parse_args()
    return pars

def load_model_weights(model, checkpoint_path):
    ckp = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckp)
    return model

def create_checkpoint_folder(checkpoint_folder, _class_):
    class_checkpoint_folder = os.path.join(checkpoint_folder, _class_)
    os.makedirs(class_checkpoint_folder, exist_ok=True)
    return class_checkpoint_folder

def train(_class_, pars):
    print(_class_)
    columns = ['epoch', 'area_leafs', 'area_leafs_gt', 'area_anomaly', 'area_anomaly_gt', 'percentage', 'percentage_gt']

    device = 'cuda' if torch.cuda.is_available() else 'cuda'

    data_transform, gt_transform, gt_leaf_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    train_path = "../../../../Data/" + _class_ + '/train'
    test_path = "../../../../Data/" + _class_
    
    if not os.path.exists(pars.save_folder + '/' + _class_):
        os.makedirs(pars.save_folder + '/' + _class_)
    save_model_path  = pars.save_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    train_data = MVTecDataset_train(root=train_path, transform=data_transform,  color_space = pars.color_space, used_noise = pars.noise)
    test_data = MVTecDataset_test(root=test_path, transform=data_transform,  gt_transform=gt_transform, gt_leaf_transform = gt_leaf_transform,  color_space = pars.color_space)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=pars.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    # Load the model 
    if pars.use_pretrained == "True":
        print("Loaded pretrained model.")

         # Use pretrained wide_resnet50 for encoder
        encoder, bn = wide_resnet50_2(pretrained=True)
        encoder = encoder.to(device)
        encoder.eval()

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
    else:
        print("Created new model without pre-trained weights.")

        # Use pretrained ImageNet for encoder
        encoder, bn = wide_resnet50_2(pretrained=True)
        encoder = encoder.to(device)
        encoder.eval()

        bn = bn.to(device)
        encoder.eval()

        decoder = de_wide_resnet50_2(pretrained=False)
        decoder = decoder.to(device)

        proj_layer =  MultiProjectionLayer(base=64).to(device)
        print("Created new model without pre-trained weights.")

    
    proj_loss = Revisit_RDLoss()
    optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=pars.proj_lr, betas=(0.5,0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=pars.distill_lr, betas=(0.5,0.999))

    
    best_score = 0
    best_epoch = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0
    
    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []
    
    loss_proj = []
    loss_distill = []
    total_loss = []
    
    history_infor = {}


    # set appropriate epochs for specific classes (Some classes pthverge faster than others)
    
    if _class_ in ['carpet','leather']:
        num_epoch = 10
    if _class_ in ['grid','tile']:
        num_epoch = 260
    if _class_ in ['wood']:
        num_epoch = 100   
    if _class_ in ['cable']:
        num_epoch = 240
    if _class_ in ['capsule']:
        num_epoch = 300
    if _class_ in ['hazelnut']:
        num_epoch = 160
    if _class_ in ['metal_nut']:
        num_epoch = 160
    if _class_ in ['screw']:
        num_epoch = 280
    if _class_ in ['toothbrush']:
        num_epoch = 280
    if _class_ in ['transistor']:
        num_epoch = 300  
    if _class_ in ['zipper']:
        num_epoch = 300
    if _class_ in ['pill']:
        num_epoch = 200
    if _class_ in ['bottle']:
        num_epoch = 200
    if _class_ in ['data_masked']:
        num_epoch = 350

    print(f'with class {_class_}, Training with {num_epoch} Epoch')
    # Initialize an empty DataFrame before the loop
    result_df = pd.DataFrame(columns=['epoch','area_leafs', 'area_leafs_gt', 'area_anomaly', 'area_anomaly_gt', 'percentage', 'percentage_gt'])
    training_df = pd.DataFrame(columns=['epoch', "total_loss", "loss_proj", "loss_distill", "auroc_px", "auroc_sp", "aupro_px"])
    for epoch in tqdm(range(1,num_epoch+1)):
        bn.train()
        proj_layer.train()
        decoder.train()
        loss_proj_running = 0
        loss_distill_running = 0
        total_loss_running = 0
        
        ## gradient acc
        accumulation_steps = 2
        
        for i, (img,img_noise,_) in enumerate(train_dataloader):
            img = img.to(device)
            img_noise = img_noise.to(device)
             
            #img_noisy = cv2.cvtColor(img_noise.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            #img_noisy = np.uint8(min_max_norm(img_noisy) * 255)
            #cv2.imwrite('./results_all/' + _class_ + '/' + str(000)+ '.png', img_noisy)

            inputs = encoder(img)
            inputs_noise = encoder(img_noise)

            (feature_space_noise, feature_space) = proj_layer(inputs, features_noise = inputs_noise)

            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)

            outputs = decoder(bn(feature_space))#bn(inputs))
            L_distill = loss_fucntion(inputs, outputs)
            loss = L_distill + pars.weight_proj * L_proj
            optimizer_proj.zero_grad()
            optimizer_distill.zero_grad()
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer_proj.step()
                optimizer_distill.step()

            total_loss_running += loss.detach().cpu().item()
            loss_proj_running += L_proj.detach().cpu().item()
            loss_distill_running += L_distill.detach().cpu().item()
            
            
        area_leafs, area_leafs_gt, area_anomaly, area_anomaly_gt, auroc_px, auroc_sp, aupro_px,f1, precision = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device, _class_, pars.color_space)       
        # Convert the dictionaries to DataFrames
        area_leafs_df = pd.DataFrame(area_leafs.values(), columns=['area_leafs'], index=area_leafs.keys())
        area_leafs_gt_df = pd.DataFrame(area_leafs_gt.values(), columns=['area_leafs_gt'], index=area_leafs_gt.keys())
        area_anomaly_df = pd.DataFrame(area_anomaly.values(), columns=['area_anomaly'], index=area_anomaly.keys())
        area_anomaly_gt_df = pd.DataFrame(area_anomaly_gt.values(), columns=['area_anomaly_gt'], index=area_anomaly_gt.keys())

        # Concatenate the DataFrames horizontally
        epoch_df = pd.concat([area_leafs_df, area_leafs_gt_df, area_anomaly_df, area_anomaly_gt_df], axis=1)

        # Calculate the percentages
        epoch_df['percentage'] = epoch_df['area_anomaly'] / epoch_df['area_leafs'] * 100
        epoch_df['percentage_gt'] = epoch_df['area_anomaly_gt'] / epoch_df['area_leafs_gt'] * 100

        # Add the epoch number to the epoch_df
        epoch_df['epoch'] = epoch

        # Reorder columns to put 'epoch' at the start
        epoch_df = epoch_df[columns]

        # Append the epoch_df to the result_df
        result_df = pd.concat([result_df, epoch_df], ignore_index=True)
        training_df.loc[epoch] = [epoch, total_loss_running, loss_proj_running, loss_distill_running, auroc_px, auroc_sp, aupro_px]

        # Print the result
        print(epoch_df)
        print(result_df)
        print(training_df)
        file_path = pars.checkpoint_folder + '/' + _class_ + '/' + 'training_df.txt'
        with open(file_path, "w") as f:
            f.write(training_df.to_string())
        file_path = pars.checkpoint_folder + '/' + _class_ + '/' + 'result_df.txt'
        with open(file_path, "w") as f:
            f.write(training_df.to_string())
                     
        auroc_px_list.append(auroc_px)
        auroc_sp_list.append(auroc_sp)
        aupro_px_list.append(aupro_px)
        loss_proj.append(loss_proj_running)
        loss_distill.append(loss_distill_running)
        total_loss.append(total_loss_running)        
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 12)
        fig, ax = plt.subplots(3,2, figsize = (8, 12))
        ax[0][0].plot(auroc_px_list)
        ax[0][0].set_title('auroc_px')
        ax[0][1].plot(auroc_sp_list)
        ax[0][1].set_title('auroc_sp')
        ax[1][0].plot(aupro_px_list)
        ax[1][0].set_title('aupro_px')
        ax[1][1].plot(loss_proj)
        ax[1][1].set_title('loss_proj')
        ax[2][0].plot(loss_distill)
        ax[2][0].set_title('loss_distill')
        ax[2][1].plot(total_loss)
        ax[2][1].set_title('total_loss')
        plt.savefig(pars.save_folder + '/' + _class_ + '/monitor_traning.png', dpi = 100)
    
        
        print('Epoch {}, Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(epoch, auroc_sp, auroc_px, aupro_px))
        torch.save({'proj': proj_layer.state_dict(),
                       'decoder': decoder.state_dict(),
                        'bn':bn.state_dict()}, save_model_path)

        if (auroc_px + auroc_sp + aupro_px) / 3 > best_score:
            best_score = (auroc_px + auroc_sp + aupro_px) / 3
            
            best_auroc_px = auroc_px
            best_auroc_sp = auroc_sp
            best_aupro_px = aupro_px
            best_epoch = epoch

            torch.save({'proj': proj_layer.state_dict(),
                       'decoder': decoder.state_dict(),
                        'bn':bn.state_dict()}, save_model_path)

            history_infor['auroc_sp'] = best_auroc_sp
            history_infor['auroc_px'] = best_auroc_px
            history_infor['aupro_px'] = best_aupro_px
            history_infor['epoch'] = best_epoch
            with open(os.path.join(pars.save_folder + '/' + _class_, f'history.json'), 'w') as f:
                json.dump(history_infor, f)
    return best_auroc_sp, best_auroc_px, best_aupro_px, total_loss, loss_proj, loss_distill, auroc_px_list, auroc_sp_list, aupro_px_list




if __name__ == '__main__':
    pars = get_args()
    all_classes = [ 'data_masked']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    
    # train all_classes
    # for c in all_classes
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px, total_loss, loss_proj, loss_distill, auroc_px_list, auroc_sp_list, aupro_px_list = train(c, pars)
        print('Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(c, auroc_sp, auroc_px, aupro_px))
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        pd.DataFrame(metrics).to_csv(f'{pars.save_folder}/metrics_results.csv', index=False)
