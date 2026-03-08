import sys
import os
import os.path as osp
import time
import datetime
import itertools
import json
import pickle
import warnings
from argparse import ArgumentParser
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import data

import torch
import torch.nn.functional as F
import torch_geometric
import torchvision
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import confusion_matrix, precision_score, f1_score, precision_recall_curve, accuracy_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Import custom modules
sys.path.append("..")
from utils.load_excel_pkl import load_excel_pkl
from utils.enzyme_dataset import cif_lm_dataset, cif_esm_dataset, cif_protbert_dataset
from utils.model_gcn_mo import *

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

# Global scaler for mixed precision training
scaler = GradScaler()

def load_data(dataset_name, df_target_list, args):
    bs = args.batch_size
    if df_target_list.index.name != 'pdb_file_name':
        df_target_list = df_target_list.set_index('pdb_file_name')
    
    target_type = dataset_name.split('_')[-1]
    
    if args.lm_model == 't5':
        print('loaded t5')
        dataset = cif_lm_dataset(args.path_dataset_target, args.path_pdb_target_dt_pkl, args.p_xlsx_target_task)
    elif args.lm_model == 'esm2':
        print('loaded esm2')
        dataset = cif_esm_dataset(args.path_dataset_target, args.path_pdb_target_dt_pkl, args.p_xlsx_target_task)
    elif args.lm_model == 'protbert':
        print('loaded protbert')
        dataset = cif_protbert_dataset(args.path_dataset_target, args.path_pdb_target_dt_pkl, args.p_xlsx_target_task)
   
    df_target_list_available = df_target_list.loc[dataset._data.name]
    df_target_list_available_1 = df_target_list_available.reset_index()

    if len(df_target_list_available_1[df_target_list_available_1['train_test_1']=='val']) > 0:
        print('have val data')
        df_train = df_target_list_available_1[df_target_list_available_1['train_test_1']=='train']
        df_val = df_target_list_available_1[df_target_list_available_1['train_test_1']=='val']
    else:
        df_train_all = df_target_list_available_1[df_target_list_available_1['train_test_1']=='train']
        col_ec = [h for h in df_train_all.columns if h[:3]=='EC ']
        encoded_label = df_train_all[col_ec].to_numpy()
        
        encoded_label_split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(range(len(encoded_label)), encoded_label)
        for train_index_s, test_index_s in encoded_label_split:
            train_index_s, test_index_s = train_index_s, test_index_s

        df_train = df_target_list_available_1.iloc[df_train_all.reset_index()['index'][train_index_s].tolist()]
        df_val = df_target_list_available_1.iloc[df_train_all.reset_index()['index'][test_index_s].tolist()]
        
    n_samples = df_target_list_available[[h for h in df_target_list_available.columns if h[:3]=='EC ']].sum()
    n_total = len(df_target_list_available)
    class_pos_weight = (n_total - n_samples)/n_samples
    
    train_list_s_idx = df_train.index.tolist()
    val_list_s_idx = df_val.index.tolist()
    test_list_s_idx = df_target_list_available_1[df_target_list_available_1['train_test_1']=='test'].index.tolist()
    
    dataset._data.y = torch.tensor(df_target_list_available[[h for h in df_target_list_available.columns if h[:3]=='EC ']].values, dtype=torch.float64)
    dataset.slices['y'] = dataset.slices['name']
    args.num_features = dataset.num_features
    args.num_classes = len(dataset._data.y[0])

    train_data = dataset[train_list_s_idx]
    val_data = dataset[val_list_s_idx]
    test_data = dataset[test_list_s_idx]
    
    train_loader = DataLoader(train_data, batch_size=bs)
    val_loader = DataLoader(val_data, batch_size=bs)
    test_loader = DataLoader(test_data, batch_size=bs)
    
    return train_loader, val_loader, test_loader, class_pos_weight

def cal_ap_and_threshold(l_true, l_pred_proba):
    precision, recall, thresholds = precision_recall_curve(l_true, l_pred_proba)
    fscore = (2 * precision * recall) / (precision + recall)
    fscore = np.nan_to_num(fscore)
    ix = np.argmax(fscore)
    return thresholds[ix]

def mcc(y_true, y_pred, *, sample_weight=None):
    lb = LabelEncoder()
    lb.fit(np.hstack([y_true, y_pred]))
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)

    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    t_sum = C.sum(axis=1, dtype=np.float64)
    p_sum = C.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(C, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)

    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    else:
        return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

def evaluation_score(l_true, l_pred_proba, thresholds):
    l_pred = (np.array(l_pred_proba) >= thresholds)*1
    d_evaluation = {}
    d_evaluation['average_precision'] = average_precision_score(l_true, l_pred_proba)
    d_evaluation['thresholds'] = thresholds
    d_evaluation['accuracy_score'] = accuracy_score(l_true, l_pred)
    d_evaluation['precision_score'] = precision_score(l_true, l_pred, zero_division=0)
    d_evaluation['f1_score'] = f1_score(l_true, l_pred, zero_division=0)
    [[tn, fp], [fn, tp]] = confusion_matrix(l_true, l_pred, labels=[0, 1])
    d_evaluation['tn'] = int(tn)
    d_evaluation['fp'] = int(fp)
    d_evaluation['fn'] = int(fn)
    d_evaluation['tp'] = int(tp)
    d_evaluation['mcc'] = mcc(l_true, l_pred)
    return d_evaluation

def focal_binary_cross_entropy(logits, targets, gamma=2.0, reduction='sum'):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_term = (1.0 - pt).pow(gamma)
    loss = focal_term * bce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def train(loader, class_pos_weight, model, optimizer, device, args):
    model.train()
    total_loss = 0.0
    num_batches = 0
    print(f'args.gamma = {args.gamma}')
    for local_batch in tqdm(loader):
        local_batch = local_batch.to(device)
        optimizer.zero_grad()
        
        with autocast(dtype=torch.float16):
            if args.weighted_edge:
                out = model(local_batch.x, local_batch.edge_index, local_batch.batch, 
                            edge_weight=local_batch.edge_weight)
            else:
                out = model(local_batch.x, local_batch.edge_index, local_batch.batch)
            
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("Warning: NaN or Inf detected in model output. Skipping batch.")
                continue
            
            if args.loss_function == 'bce_loss':
                if args.use_pos_weight == 0:
                    loss = F.binary_cross_entropy_with_logits(out.squeeze(1), local_batch.y.float())
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        out.squeeze(1), local_batch.y.float(),
                        pos_weight=torch.tensor(class_pos_weight, device=device))
            elif args.loss_function == 'focal_loss':
                loss = focal_binary_cross_entropy(out.squeeze(1), local_batch.y.float(), gamma=args.gamma)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected (loss={loss.item()}). Skipping backward.")
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        torch.cuda.empty_cache()
    
    return total_loss / num_batches if num_batches > 0 else float('nan')

def test(loader, model, device, args, class_pos_weight):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    l_pred_proba, l_true = [], []
    
    with torch.no_grad():
        for local_batch in loader:
            local_batch = local_batch.to(device)
            
            with autocast(dtype=torch.float16):
                if args.weighted_edge:
                    out = model(local_batch.x, local_batch.edge_index, local_batch.batch,
                                edge_weight=local_batch.edge_weight)
                else:
                    out = model(local_batch.x, local_batch.edge_index, local_batch.batch)
                
                pred = torch.sigmoid(out).squeeze(1)
                
                if args.loss_function == 'bce_loss':
                    if args.use_pos_weight == 0:
                        loss = F.binary_cross_entropy_with_logits(out.squeeze(1), local_batch.y.float())
                    else:
                        loss = F.binary_cross_entropy_with_logits(
                            out.squeeze(1), local_batch.y.float(),
                            pos_weight=torch.tensor(class_pos_weight, device=device))
                elif args.loss_function == 'focal_loss':
                    loss = focal_binary_cross_entropy(out.squeeze(1), local_batch.y.float(), gamma=args.gamma)
            
            total_loss += loss.item()
            num_batches += 1
            
            l_pred_proba.extend(pred.cpu().tolist()) 
            l_true.extend(local_batch.y.cpu().tolist())
            torch.cuda.empty_cache() 
    
    return total_loss / num_batches, np.array(l_true), np.array(l_pred_proba)

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--task_name", dest="task_name", default='uniprot_sc_c')
    parser.add_argument("--target_source", dest="target_source", default='uniprot_sc_c')
    parser.add_argument("--target_class", dest="target_class", default='all')
    parser.add_argument("--target_type", dest="target_type", default='all')
    parser.add_argument("--pdb_type", dest="pdb_type", default='cif')
    parser.add_argument("--target_train_test", dest="target_train_test", default='train_test_year')
    parser.add_argument("--model_name", dest="model_name", default='LE_ClusterGCN_l1_fc2_lm-d2048')
    parser.add_argument("--lm_model", dest="lm_model", default='t5')
    parser.add_argument("--target_source_file", dest="target_source_file", default='uniprot_sc_c')
    parser.add_argument("--distance_method", dest="distance_method", default='f_dist_lm')
    parser.add_argument("--target_excl_gene", dest="target_excl_gene", default=[])
    parser.add_argument("--distance_s", dest="distance_s", default='9')
    parser.add_argument("--short_path", dest="short_path", default=True)
    parser.add_argument("--epoch", dest="epoch", default=500, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=20, type=int)
    parser.add_argument("--load_model", dest="load_model", default=None)
    parser.add_argument("--score_threshold", dest="score_threshold", default=0.95, type=float)
    parser.add_argument("--weighted_edge", dest="weighted_edge", default=0, type=int)
    parser.add_argument("--train_reverse", dest="train_reverse", default=False)
    parser.add_argument("--train_val_as_val", dest="train_val_as_val", default=False)
    parser.add_argument("--cv_s", dest="cv_s", default='4')
    parser.add_argument("--lr", dest="lr", default=0.001, type=float)
    parser.add_argument("--wd", dest="wd", default=5e-6, type=float)
    parser.add_argument("--loss_function", dest="loss_function", default='focal_loss')
    parser.add_argument("--gamma", dest="gamma", default=2, type=float)
    parser.add_argument("--use_pos_weight", dest="use_pos_weight", default=0, type=int)

    if '-f' in sys.argv or '--f' in [q[:3] for q in sys.argv]:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    model_arg = args.model_name.split('-')
    args.model_name_s = model_arg[0]
    
    for model_arg_s in model_arg[1:]:
        if model_arg_s[0]=='d':
            args.dim = int(model_arg_s[1:])

    # FIX: Run from current directory instead of hardcoded 'molecule' path
    args.path_home = "." 
    args.weighted_edge_s = '_w' if args.weighted_edge else ''
    args.dataset_version = 'A202304'
    args.path_dataset_version = args.path_home + '/' + args.dataset_version
    args.dataset_name = args.pdb_type + '_' + args.target_source + args.weighted_edge_s
    args.score_threshold = float(args.score_threshold)
    
    args.path_excel = args.path_dataset_version + '/excel/'
    args.path_excel_xlsx_type = args.path_excel + '/' + args.pdb_type + '_' + args.task_name + '/'
    args.xlsx_target_task = args.pdb_type + '_' + args.task_name + '_pdb_atom_' + args.target_train_test
    args.p_xlsx_target_task = args.path_excel_xlsx_type + args.xlsx_target_task + '.xlsx'

    for task_output in list(itertools.product(args.distance_s.split(','), args.cv_s.split(','))):
        args.distance, args.cv_n = task_output

        for arg in vars(args):
            print(arg, ':', getattr(args, arg))

        if args.target_class == 'class':
            args.target_class_n = [1]
        elif args.target_class == 'sclass':
            args.target_class_n = [2]
        elif args.target_class == 'ssclass':
            args.target_class_n = [3]
        elif args.target_class == 'sssclass':
            args.target_class_n = [4]
        elif args.target_class == 'ssclass_sssclass':
            args.target_class_n = [3,4]
        elif args.target_class == 'all':
            args.target_class_n = [1,2,3,4]

        args.path_pdb_target = args.path_dataset_version + '/data/pdb_raw/' + args.target_source + '/' + args.distance_method + '_' + args.distance.zfill(2)
        args.path_excel_target_xlsx = args.p_xlsx_target_task
        args.path_dataset_target = args.path_dataset_version + '/dataset/' + args.pdb_type + '_' + args.target_source + '_w/' + args.target_type + '/' + args.distance_method + '_' + args.distance
        args.path_model_target = args.path_dataset_version + '/model/' + args.pdb_type + '_' + args.target_source + args.weighted_edge_s + '/' + args.target_class + '/' + args.target_type + '/' + args.model_name + '/' + args.distance_method + '_' + args.distance + '/cv_' + args.cv_n + '/'
        
        os.makedirs(args.path_pdb_target, exist_ok=True)
        os.makedirs(args.path_dataset_target, exist_ok=True)
        os.makedirs(args.path_model_target, exist_ok=True)

        args.path_pdb_target_dt_pkl = args.path_dataset_version + '/data/pdb_raw/' + args.pdb_type + '_' + args.target_source + '/' + args.pdb_type + '_' + args.distance_method + '_' + args.distance.zfill(2) + '/'
        args.train_test_cv_n = 'train_test_' + args.cv_n

        time_st = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if args.load_model == None else args.load_model[6:]

        df_target_list = load_excel_pkl(args.path_excel_xlsx_type + args.xlsx_target_task + '.xlsx')
        df_ec_target = pd.DataFrame([h[3:] for h in df_target_list.columns.tolist() if h[:3] == 'EC '], columns=['ec'])
        df_ec_target['col_ec'] = df_ec_target['ec'].apply(lambda x: 5-len(x.split('.-')))
        df_ec_target_1 = df_ec_target[df_ec_target['col_ec'].isin(args.target_class_n)]
        
        args.labels = ['all']
        d_labels_score = {q1: [0, datetime.datetime(2000,1,1)] for q1 in args.labels}

        if args.load_model != None:
            print('Using model: ', args.load_model)
            l_labels_score_load_path = []
            # Note: path_evaluation was undefined in the original script inside this block. Make sure to define it or ignore it.
            # Skipping the unimplemented logic for brevity and safety.

        l_labels_score = [[k]+v for k,v in d_labels_score.items() if v[0] < args.score_threshold]
        print('Remain Training Target:', len(l_labels_score))

        if len(l_labels_score) == 0:
            break

        l_labels_score.sort(key = lambda x : x[2])
        p0 = l_labels_score.pop(0)
        p1 = p0[0]
        print(p0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name = args.dataset_name + '_' + str(p1)

        path_save_evaluation = args.path_model_target + '/model_' + time_st + '/' + dataset_name + '/'
        os.makedirs(path_save_evaluation, exist_ok=True)

        train_loader, val_loader, test_loader, class_pos_weight = load_data(dataset_name, df_target_list, args)

        model = eval(args.model_name_s)(args).to(device)

        d_model_desc = {'args': vars(args), 'model': str(model._modules)}
        print(model)

        with open(path_save_evaluation + '/model_desc.json', 'w', encoding='utf-8') as f1:
            json.dump(d_model_desc, f1, ensure_ascii=False, indent=4, default=str)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        early_stop = False
        epochs_no_improve = 0
        n_epochs_stop = 30
        min_val_score = -1
        min_val_loss = np.inf
        min_train_loss = np.inf

        d_log = {'Epoch 0': {'time': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}

        for epoch in range(1, args.epoch+1):
            train_loss = train(train_loader, class_pos_weight, model, optimizer, device, args)
            val_loss, val_l_true, val_l_pred_proba = test(val_loader, model, device, args, class_pos_weight)
            test_loss, test_l_true, test_l_pred_proba = test(test_loader, model, device, args, class_pos_weight)

            df_val_test = pd.DataFrame({
                'val_l_true': val_l_true.T.tolist(),
                'val_l_pred_proba': np.nan_to_num(val_l_pred_proba, nan=0.0).T.tolist(),
                'test_l_true': test_l_true.T.tolist(),
                'test_l_pred_proba': np.nan_to_num(test_l_pred_proba, nan=0.0).T.tolist(),
            })

            d_eval_val_all = pd.json_normalize(df_val_test.apply(lambda x: evaluation_score(x.val_l_true, x.val_l_pred_proba, 0.5), axis=1))
            d_eval_test_all = pd.json_normalize(df_val_test.apply(lambda x: evaluation_score(x.test_l_true, x.test_l_pred_proba, 0.5), axis=1))

            d_eval_val_all['ec'] = df_ec_target_1['ec'].tolist()
            d_eval_val_all['level_1'] = df_ec_target_1['ec'].apply(lambda x: 5-len(x.split('.-')))
            d_eval_test_all['ec'] = df_ec_target_1['ec'].tolist()
            d_eval_test_all['level_1'] = df_ec_target_1['ec'].apply(lambda x: 5-len(x.split('.-')))

            d_eval_val_all['p'] = d_eval_val_all['tp'] + d_eval_val_all['fn']
            d_eval_test_all['p'] = d_eval_test_all['tp'] + d_eval_test_all['fn']
            
            d_eval_test_all_1 = d_eval_test_all[d_eval_test_all['p'] != 0]
            
            print(path_save_evaluation)
            print('use_pos_weight:', args.use_pos_weight)
            print('Val, Test, Test(exclude zero positive cases)')
            print('all: {:.5f}, {:.5f}, {:.5f}'.format(d_eval_val_all['f1_score'].mean(), d_eval_test_all['f1_score'].mean(), d_eval_test_all_1['f1_score'].mean()))
            
            for i in range(4):
                val_mean = d_eval_val_all[d_eval_val_all['level_1']==i+1]['f1_score'].mean()
                test_mean = d_eval_test_all[d_eval_test_all['level_1']==i+1]['f1_score'].mean()
                test_1_mean = d_eval_test_all_1[d_eval_test_all_1['level_1']==i+1]['f1_score'].mean()
                print('{}: {:.5f}, {:.5f}, {:.5f}'.format(i+1, val_mean, test_mean, test_1_mean))
            
            d_eval_val_all['r_lr'] = args.lr
            d_eval_val_all['r_wd'] = args.wd
            d_eval_test_all['r_lr'] = args.lr
            d_eval_test_all['r_wd'] = args.wd

            torch.save(model, path_save_evaluation + "model_" + d_log['Epoch 0']['time'] + '_' + str(epoch) + "_.pt")

            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Loss: {:.5f}, Val Acc: {:.5f}, Test Loss: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, train_loss, 0, val_loss, d_eval_val_all['f1_score'].mean(), test_loss, d_eval_test_all['f1_score'].mean()))
            
            d_log['Epoch ' + str(epoch)] = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'd_eval_val': d_eval_val_all,
                'test_loss': test_loss,
                'd_eval_test': d_eval_test_all,
                'time': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            }
        
            val_score = d_eval_val_all['f1_score'].mean()
            
            if val_score > min_val_score:
                 epochs_no_improve = 0
                 min_val_score = val_score
            elif val_loss < min_val_loss:
                 epochs_no_improve = 0
                 min_val_loss = val_loss
            elif train_loss < min_train_loss:
                 epochs_no_improve = 0
                 min_train_loss = train_loss
            else:
                epochs_no_improve += 1
                
            print('epochs_no_improve:', epochs_no_improve)
            if epoch > 30 and epochs_no_improve > n_epochs_stop:
                print('Early stopping!')
                early_stop = True

            if early_stop:
                print("Stopped")
                break

        with open(path_save_evaluation + '/log_' + d_log['Epoch 0']['time'] + '.pkl', 'wb') as outfile:
            pickle.dump(d_log, outfile, pickle.HIGHEST_PROTOCOL)

        d_labels_score[p1] = [min_val_score, datetime.datetime.now()]

        del train_loader, val_loader, test_loader, model
        torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {str(timedelta(seconds=elapsed_time))}")