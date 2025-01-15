#@title 5. Run Prediction
import os
import json
import sys
import pickle
import pandas as pd
import argparse
from Bio import SwissProt
from argparse import ArgumentParser
import torch
import torch_geometric
import torchvision
from torch_geometric.loader import DataLoader
from Bio.Align import PairwiseAligner
from transformers import T5Tokenizer, T5EncoderModel, AutoModel
from tqdm import tqdm
import datetime
from collections import Counter
import wget
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.algorithm import CaptumExplainer
from torch_geometric.explain.config import (
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)

from scipy.sparse import csr_matrix, triu
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
import re
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import math
import numpy as np
from sklearn import preprocessing
import datetime
os.umask(0)
from sklearn.preprocessing import MinMaxScaler
from utils.model_gcn_mo import *


def load_pdb_data(args,bs):
    


    codification = { "ALA" : 'A',
                    "CYS" : 'C',
                    "ASP" : 'D',
                    "GLU" : 'E',
                    "PHE" : 'F',
                    "GLY" : 'G',
                    "HIS" : 'H',
                    "ILE" : 'I',
                    "LYS" : 'K',
                    "LEU" : 'L',
                    "MET" : 'M',
                    "ASN" : 'N',
                    # "PYL" : 'O',
                    "PRO" : 'P',
                    "GLN" : 'Q',
                    "ARG" : 'R',
                    "SER" : 'S',
                    "THR" : 'T',
                    # "SEC" : 'U',
                    "VAL" : 'V',
                    "TRP" : 'W',
                    "TYR" : 'Y',
                    # "UNK" : 'X'
                    }

    l_amino_acid = list(codification.keys())
        
    l_amino_acid_all = l_amino_acid 


    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l_amino_acid_all)

    onehot_encoder = preprocessing.OneHotEncoder(sparse_output=False,categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded) 
    
    
    target_distance = 9
    
    path = args.target_source
    


    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    # print("Loading: {}".format(transformer_link))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_lm = T5EncoderModel.from_pretrained(transformer_link)
    model_lm.full() if device=='cpu' else model_lm.half() # only cast to full-precision if no GPU is available
    
    model_lm = model_lm.to(device)
    model_lm = model_lm.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )




    data_list = []
    target_files_pdb = [h for h in os.listdir(args.target_source) if h.split('.')[-1]=='cif']
    print(target_files_pdb)
    for pdb_files in target_files_pdb:

        pdb_name_s = pdb_files.split('.')[0]

        

        with open(args.target_source+'/'+pdb_name_s+'.cif','r') as f1:
            data = f1.readlines()

        section_name = None
        s = 0
        cif_target_list = ['_atom_site','_struct_ref_seq']

        d_cif = {}
        for h1 in cif_target_list:
            d_cif[h1] = []
            d_cif[h1+'_col'] = []


        is_table=False

        for h in data:
            
            if h in ['# \n','#\n']:
                section_name = None
                is_table = False
            elif h == 'loop_\n':
                is_table = True
            else:
                if section_name == None:
                    section_name = h.split('.')[0]
                    d_cif[section_name+'_is_table']=is_table
                    
            
                if section_name in cif_target_list:
                    
                    len_section_name = len(section_name)
                    l_data = list(filter(None, h[:-1].split(' ')))
                    
                    if is_table:
                        if l_data[0][:len_section_name]==section_name:
                            d_cif[section_name+'_col']+=[l_data[0][len_section_name+1:]]
                        else:
                            d_cif[section_name]+=[l_data]
                    else:
                        
                        d_cif[section_name]+=[l_data[1]]
                        d_cif[section_name+'_col']+=[l_data[0][len_section_name+1:]]

        df = pd.DataFrame(d_cif['_atom_site'],columns=d_cif['_atom_site_col'])
        chain_all = df['label_asym_id'].unique().tolist()

        
        
        
        if d_cif['_struct_ref_seq_is_table']:
            df_dbref = pd.DataFrame(d_cif['_struct_ref_seq'],columns=d_cif['_struct_ref_seq_col'])

        else:
            df_dbref = pd.DataFrame([d_cif['_struct_ref_seq']],columns=d_cif['_struct_ref_seq_col'])
        
        
        
        with open(args.target_source+'/'+pdb_name_s+'.cif','r') as f1:
            data = f1.readlines()
            
            
        doc = []
        col_atom_site = []
        s = 0
        for h in data:

            if h[:11]=='_atom_site.':
                col_atom_site += [h[11:].replace(' ','').replace('\n','')]
                s = 1
            elif s==1 and h[:5]=='ATOM ':
                doc += [list(filter(None, h[:-1].split(' ')))]
            elif s==1 and h=='# \n':
                s=0
    
    
        df_atoms_o = df[df['group_PDB']=='ATOM'].reset_index(drop=True)
        
        pdb_name_chain_s = df_atoms_o['auth_asym_id'].unique()
    

        for pdb_name_chain in pdb_name_chain_s:
            
            df_atoms = df_atoms_o.copy()
                
            model_num = df_atoms['pdbx_PDB_model_num'].iloc[0]

            df_atoms = df_atoms[(df_atoms['group_PDB']=='ATOM')&(df_atoms['pdbx_PDB_model_num']==model_num)&(df_atoms['auth_asym_id']==pdb_name_chain)].reset_index(drop=True)
            
            
            df_atoms = df_atoms[df_atoms['auth_comp_id'].isin(codification.keys())].reset_index(drop=True)
    
        
            df_atoms['auth_comp_id_code'] = df_atoms['auth_comp_id'].apply(lambda x: codification[x])

            df_atoms['chainID_resSeq'] = df_atoms['auth_asym_id'] + df_atoms['auth_seq_id'].str.zfill(5)

            # df_atoms = df_atoms[(df_atoms['label_alt_id']=='.')|(df_atoms['label_alt_id']=='A')].reset_index(drop=True)

            df_atoms = df_atoms.drop_duplicates(['chainID_resSeq','auth_atom_id']).reset_index(drop=True)
            


            df_atoms['Cartn_x'] = df_atoms['Cartn_x'].apply(float)
            df_atoms['Cartn_y'] = df_atoms['Cartn_y'].apply(float)
            df_atoms['Cartn_z'] = df_atoms['Cartn_z'].apply(float)



            df_atoms['Cartn_x_CA'] = df_atoms['Cartn_x']
            df_atoms['Cartn_y_CA'] = df_atoms['Cartn_y']
            df_atoms['Cartn_z_CA'] = df_atoms['Cartn_z'] 
            df_atoms['Cartn_n_CA'] = 1

            df_atoms.loc[~(df_atoms['auth_atom_id']=='CA'), ['Cartn_x_CA','Cartn_y_CA','Cartn_z_CA','Cartn_n_CA']] = 0
            

            df_atoms['Cartn_x_CB'] = df_atoms['Cartn_x']
            df_atoms['Cartn_y_CB'] = df_atoms['Cartn_y']
            df_atoms['Cartn_z_CB'] = df_atoms['Cartn_z'] 
            df_atoms['Cartn_n_CB'] = 1

            df_atoms.loc[~(df_atoms['auth_atom_id']=='CB'), ['Cartn_x_CB','Cartn_y_CB','Cartn_z_CB','Cartn_n_CB']] = 0
            
            df_atoms['Cartn_x_CE'] = df_atoms['Cartn_x']
            df_atoms['Cartn_y_CE'] = df_atoms['Cartn_y']
            df_atoms['Cartn_z_CE'] = df_atoms['Cartn_z']






            df_1 = df_atoms.groupby('chainID_resSeq').agg({'auth_comp_id':'first','group_PDB':'count'
                                                        ,'Cartn_x_CA':'sum','Cartn_y_CA':'sum','Cartn_z_CA':'sum','Cartn_n_CA':'sum'
                                                        ,'Cartn_x_CB':'sum','Cartn_y_CB':'sum','Cartn_z_CB':'sum','Cartn_n_CB':'sum'
                                                        ,'Cartn_x_CE':'sum','Cartn_y_CE':'sum','Cartn_z_CE':'sum'})
            

            df_1['group_PDB'] = df_1['group_PDB'].apply(float)

            # if len(df_1[df_1['Cartn_n_CA']!=1]):
            #     print('CA atom not found:',pdb_name,pdb_name_chain,df_1[df_1['Cartn_n_CA']!=1].index.tolist())
                # df_atoms[df_atoms['chainID_resSeq']=='A00393']

            df_1['Cartn_x_CE'] = df_1['Cartn_x_CE']/df_1['group_PDB']
            df_1['Cartn_y_CE'] = df_1['Cartn_y_CE']/df_1['group_PDB']
            df_1['Cartn_z_CE'] = df_1['Cartn_z_CE']/df_1['group_PDB']


            df_1 = df_1.drop('group_PDB',axis=1)


            df_1['Cartn_x'] = df_1['Cartn_x_CA']
            df_1['Cartn_y'] = df_1['Cartn_y_CA']
            df_1['Cartn_z'] = df_1['Cartn_z_CA']


            df_1.loc[~(df_1['Cartn_n_CA']==1), ['Cartn_x']] = df_1['Cartn_x_CB']
            df_1.loc[~(df_1['Cartn_n_CA']==1), ['Cartn_y']] = df_1['Cartn_y_CB']
            df_1.loc[~(df_1['Cartn_n_CA']==1), ['Cartn_z']] = df_1['Cartn_z_CB']

            df_1['Cartn_n_CA_CB'] = df_1['Cartn_n_CA']+df_1['Cartn_n_CB']

            df_1.loc[~(df_1['Cartn_n_CA_CB']>0), ['Cartn_x']] = df_1['Cartn_x_CE']
            df_1.loc[~(df_1['Cartn_n_CA_CB']>0), ['Cartn_y']] = df_1['Cartn_y_CE']
            df_1.loc[~(df_1['Cartn_n_CA_CB']>0), ['Cartn_z']] = df_1['Cartn_z_CE']


            df_1['auth_comp_id_code'] = df_1['auth_comp_id'].apply(lambda x: codification[x])

            df_1 = df_1.reset_index()

            
            
            # df_node = df_1.reset_index()
            # df_distance = pd.DataFrame(distance_matrix(df_1[['x','y','z']].values, df_1[['x','y','z']].values))
            d1 = distance_matrix(df_1[['Cartn_x','Cartn_y','Cartn_z']].values, df_1[['Cartn_x','Cartn_y','Cartn_z']].values)


            d2 = (d1>0) & (d1 < int(args.distance))*1
            
            
            edges_sp_m = csr_matrix(d2)
            # coo_edges = coo_matrix(d2)
            # df_distance.values
            
            
            node_labels = df_1['auth_comp_id']
            node_labels_1 = onehot_encoder.transform(label_encoder.transform(node_labels).reshape(-1,1))
            # node_labels_2 = np.concatenate([node_labels_1,df_aa.to_numpy()],axis=1)


            pdb_target = {'node_attributes':node_labels_1,
                    'node_auth_comp_id_code':''.join(df_1['auth_comp_id_code'].tolist()),
                    'edge_attributes':edges_sp_m}
        

            sequence_examples = [pdb_target['node_auth_comp_id_code']]
            # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # tokenize sequences and pad up to the longest sequence in the batch
            ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            # generate embeddings
            with torch.no_grad():
                embedding_repr = model_lm(input_ids=input_ids,attention_mask=attention_mask)

            # extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7]) 
            emb_0 = embedding_repr.last_hidden_state[0,:len(pdb_target['node_auth_comp_id_code'])] # shape (7 x 1024)



            x1 = torch.from_numpy(pdb_target['node_attributes']).type(torch.FloatTensor)
            
            x = torch.cat((x1,emb_0.cpu()),1)


            y = torch.tensor([[0]], dtype=torch.float64)
            coo_edges = pdb_target['edge_attributes'].tocoo()
            edge_index = torch.tensor(np.array([coo_edges.row,coo_edges.col]), dtype=torch.int64)
            
            edge_weight = torch.tensor(np.array(coo_edges.data), dtype=torch.float32)

            # seq = torch.tensor(''.join(df_1['auth_comp_id_code'].tolist()))
            
            df_dbref_1 = df_dbref[df_dbref['pdbx_strand_id']==pdb_name_chain]
            
            data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight,name=pdb_name_s+'_'+pdb_name_chain,uniprot_id=df_dbref_1['pdbx_db_accession'].unique().tolist())
            # data = Data(x=x, y=y, edge_index=edge_index,name=target_pdb_list_s)

            # data.name = pdb_id_chain


            data_list.append(data)
        

    loader = DataLoader(data_list, batch_size=bs)
    
    

    return loader




def draw_prediction(k,v,d_config,df_uniprot_sites, df_mcsa_data):
    
    
    DRAW_UNIPROT_SITES = 1


    TARGET_WEIGHT = d_config.heatmap_method
    
    
    


    def f_1(x):
        if x == 0:
            return '  0  '
        elif x%10==0:
            return str(x)
        else:
            return ''

    def f_valid_chain(x):
        if x['valid_chain'] == True and x['valid_chain_uniprot']==True:
            return 'B'
        elif x['valid_chain'] == True:
            return 'M'
        elif x['valid_chain_uniprot']==True:
            return 'U'
        else:
            return ''

    def f_join(x):
        return '\n'.join([str(x['n']),x['sites_combined']])

    codification = { "ALA" : 'A',
                    "CYS" : 'C',
                    "ASP" : 'D',
                    "GLU" : 'E',
                    "PHE" : 'F',
                    "GLY" : 'G',
                    "HIS" : 'H',
                    "ILE" : 'I',
                    "LYS" : 'K',
                    "LEU" : 'L',
                    "MET" : 'M',
                    "ASN" : 'N',
                    # "PYL" : 'O',
                    "PRO" : 'P',
                    "GLN" : 'Q',
                    "ARG" : 'R',
                    "SER" : 'S',
                    "THR" : 'T',
                    # "SEC" : 'U',
                    "VAL" : 'V',
                    "TRP" : 'W',
                    "TYR" : 'Y',
                    # "UNK" : 'X'
                    }

    l_amino_acid = list(codification.keys())


    l_colors = ['#ced4da','#fd6104','#CC0000']
    l_colors_1 = [l_colors[0]]*18+[l_colors[1]]+[l_colors[2]]

    
    cmap = (matplotlib.colors.ListedColormap(l_colors_1))
    cmap.set_under('#FFFFFF')

    if 1:

        df_1 = pd.DataFrame(v)
        # df_1 = df_1.abs()
            
        df_1[1] = df_1[0]


        df_1 = df_1.rank()
        df_1= (df_1-df_1.min())/(df_1.max()-df_1.min())


        weights = df_1[1].to_numpy()
    
    
    
    
    
    

    pdb_name = k
    
    # pdb_no = pdb_name.split('_')[0]
    pdb_name_s = pdb_name.split('_')[0]
    pdb_chain = pdb_name.split('_')[1]
    

    p1_ec = d_config.ec.split('.-')[0]

    path_pdb = d_config.path_pdb
    path_output_img = d_config.path_output_s1+d_config.pdb_name+' '+d_config.ec+'/'
    os.makedirs(path_output_img,exist_ok=True)




    with open(path_pdb+'/'+pdb_name_s+'.cif','r') as f1:
        data = f1.readlines()

    section_name = None
    s = 0
    cif_target_list = ['_atom_site','_struct_ref_seq']

    d_cif = {}
    for h1 in cif_target_list:
        d_cif[h1] = []
        d_cif[h1+'_col'] = []


    is_table=False

    for h in data:
        
        if h in ['# \n','#\n']:
            section_name = None
            is_table = False
        elif h == 'loop_\n':
            is_table = True
        else:
            if section_name == None:
                section_name = h.split('.')[0]
                d_cif[section_name+'_is_table']=is_table
                
        
            if section_name in cif_target_list:
                
                len_section_name = len(section_name)
                l_data = list(filter(None, h[:-1].split(' ')))
                
                if is_table:
                    if l_data[0][:len_section_name]==section_name:
                        d_cif[section_name+'_col']+=[l_data[0][len_section_name+1:]]
                    else:
                        d_cif[section_name]+=[l_data]
                else:
                    
                    d_cif[section_name]+=[l_data[1]]
                    d_cif[section_name+'_col']+=[l_data[0][len_section_name+1:]]

    df = pd.DataFrame(d_cif['_atom_site'],columns=d_cif['_atom_site_col'])
    chain_all = df['label_asym_id'].unique().tolist()

    if d_cif['_struct_ref_seq_is_table']:
        df_dbref = pd.DataFrame(d_cif['_struct_ref_seq'],columns=d_cif['_struct_ref_seq_col'])

    else:
        df_dbref = pd.DataFrame([d_cif['_struct_ref_seq']],columns=d_cif['_struct_ref_seq_col'])
    
    # df_dbref['pos_diff'] = df_dbref['pdbx_auth_seq_align_beg'].apply(int)-df_dbref['db_align_beg'].apply(int)
        
    df_dbref['pos_diff'] = df_dbref['seq_align_beg'].apply(int)-df_dbref['db_align_beg'].apply(int)
    df_dbref_1 = df_dbref[df_dbref['pdbx_strand_id']==pdb_chain].reset_index(drop=True)




    model_num = df['pdbx_PDB_model_num'].iloc[0]

    df = df[(df['group_PDB']=='ATOM')&(df['pdbx_PDB_model_num']==model_num)&(df['auth_asym_id']==pdb_chain)].reset_index(drop=True)
    
    
    df = df[df['auth_comp_id'].isin(codification.keys())].reset_index(drop=True)
    if len(df)==0:
        print('emtpy data:'+pdb_name)



    df['auth_comp_id_code'] = df['auth_comp_id'].apply(lambda x: codification[x])

    df['chainID_resSeq'] = df['auth_asym_id'] + df['auth_seq_id'].str.zfill(5)

    df = df.drop_duplicates(['chainID_resSeq','auth_atom_id']).reset_index(drop=True)
    
    

    df['Cartn_x'] = df['Cartn_x'].apply(float)
    df['Cartn_y'] = df['Cartn_y'].apply(float)
    df['Cartn_z'] = df['Cartn_z'].apply(float)
    
    
    
    df['Cartn_x_CA'] = df['Cartn_x']
    df['Cartn_y_CA'] = df['Cartn_y']
    df['Cartn_z_CA'] = df['Cartn_z'] 
    df['Cartn_n_CA'] = 1

    df.loc[~(df['auth_atom_id']=='CA'), ['Cartn_x_CA','Cartn_y_CA','Cartn_z_CA','Cartn_n_CA']] = 0
    

    df['Cartn_x_CB'] = df['Cartn_x']
    df['Cartn_y_CB'] = df['Cartn_y']
    df['Cartn_z_CB'] = df['Cartn_z'] 
    df['Cartn_n_CB'] = 1

    df.loc[~(df['auth_atom_id']=='CB'), ['Cartn_x_CB','Cartn_y_CB','Cartn_z_CB','Cartn_n_CB']] = 0
    
    df['Cartn_x_CE'] = df['Cartn_x']
    df['Cartn_y_CE'] = df['Cartn_y']
    df['Cartn_z_CE'] = df['Cartn_z']






    df_1 = df.groupby('chainID_resSeq').agg({'auth_comp_id':'first','group_PDB':'count'
                                                ,'Cartn_x_CA':'sum','Cartn_y_CA':'sum','Cartn_z_CA':'sum','Cartn_n_CA':'sum'
                                                ,'Cartn_x_CB':'sum','Cartn_y_CB':'sum','Cartn_z_CB':'sum','Cartn_n_CB':'sum'
                                                ,'Cartn_x_CE':'sum','Cartn_y_CE':'sum','Cartn_z_CE':'sum'
                                                ,'label_comp_id':'first','auth_comp_id_code':'first','auth_asym_id':'first','label_seq_id':'first','auth_seq_id':'first','B_iso_or_equiv':'first'
                                                })
    

    df_1['group_PDB'] = df_1['group_PDB'].apply(float)

    if len(df_1[df_1['Cartn_n_CA']!=1]):
        print('CA atom not found:',pdb_name,df_1[df_1['Cartn_n_CA']!=1].index.tolist())
        # df[df['chainID_resSeq']=='A00393']

    df_1['Cartn_x_CE'] = df_1['Cartn_x_CE']/df_1['group_PDB']
    df_1['Cartn_y_CE'] = df_1['Cartn_y_CE']/df_1['group_PDB']
    df_1['Cartn_z_CE'] = df_1['Cartn_z_CE']/df_1['group_PDB']


    df_1 = df_1.drop('group_PDB',axis=1)


    df_1['Cartn_x'] = df_1['Cartn_x_CA']
    df_1['Cartn_y'] = df_1['Cartn_y_CA']
    df_1['Cartn_z'] = df_1['Cartn_z_CA']


    df_1.loc[~(df_1['Cartn_n_CA']==1), ['Cartn_x']] = df_1['Cartn_x_CB']
    df_1.loc[~(df_1['Cartn_n_CA']==1), ['Cartn_y']] = df_1['Cartn_y_CB']
    df_1.loc[~(df_1['Cartn_n_CA']==1), ['Cartn_z']] = df_1['Cartn_z_CB']

    df_1['Cartn_n_CA_CB'] = df_1['Cartn_n_CA']+df_1['Cartn_n_CB']

    df_1.loc[~(df_1['Cartn_n_CA_CB']>0), ['Cartn_x']] = df_1['Cartn_x_CE']
    df_1.loc[~(df_1['Cartn_n_CA_CB']>0), ['Cartn_y']] = df_1['Cartn_y_CE']
    df_1.loc[~(df_1['Cartn_n_CA_CB']>0), ['Cartn_z']] = df_1['Cartn_z_CE']


    
    
    
    
    
    # df_1 = df.groupby('chainID_resSeq').agg({'label_comp_id':'first','auth_comp_id_code':'first','group_PDB':'count','auth_asym_id':'first','label_seq_id':'first','auth_seq_id':'first','Cartn_x':'sum','Cartn_y':'sum','Cartn_z':'sum','B_iso_or_equiv':'first'})
    


    weights = minmax_scale(weights, feature_range=(0, 1), axis=0)
    
    df_1['weights_value'] = weights
    df_1['weights'] = weights

    top_n = math.ceil(len(weights)*0.05)
    # weights[np.argpartition(weights,-top_n)][-top_n:]
    top_x = weights[np.argsort(weights)[-top_n:]][0]
    top_x1 = max(top_x,1e-99)


    top_n = math.ceil(len(weights)*0.1)
    # weights[np.argpartition(weights,-top_n)][-top_n:]
    top_x = weights[np.argsort(weights)[-top_n:]][0]
    top_x2 = max(top_x,1e-99)


    # top_n = math.ceil(len(weights)*0.3)
    # # weights[np.argpartition(weights,-top_n)][-top_n:]
    # top_x = weights[np.argsort(weights)[-top_n:]][0]
    # top_x3 = max(top_x,1e-99)

    def f_weights(x,top_x1,top_x2):
        if x >= top_x1:
            return 0.95
        elif x >= top_x2:
            return 0.9
        else:
            return 0



    df_1['weights'] = df_1['weights'].apply(lambda x: f_weights(x,top_x1,top_x2))
    
    # df_1['resName_s'] = df_1['resName'].apply(lambda x:f_amino_acid(x))

    df_1['weights_+/-_1'] = df_1['weights'].rolling(2*1+1,center=True,min_periods=0).max()
    df_1['weights_+/-_2'] = df_1['weights'].rolling(2*2+1,center=True,min_periods=0).max()
    df_1['weights_+/-_3'] = df_1['weights'].rolling(2*3+1,center=True,min_periods=0).max()
    df_1['weights_+/-_4'] = df_1['weights'].rolling(2*4+1,center=True,min_periods=0).max()
    df_1['weights_+/-_5'] = df_1['weights'].rolling(2*5+1,center=True,min_periods=0).max()
    df_1['weights_+/-_6'] = df_1['weights'].rolling(2*6+1,center=True,min_periods=0).max()
    df_1['weights_+/-_7'] = df_1['weights'].rolling(2*7+1,center=True,min_periods=0).max()
    df_1['weights_+/-_8'] = df_1['weights'].rolling(2*8+1,center=True,min_periods=0).max()
    df_1['weights_+/-_9'] = df_1['weights'].rolling(2*9+1,center=True,min_periods=0).max()
    

    df_1['label_seq_id'] = df_1['label_seq_id'].apply(int)

    # df_1['resSeq_uniprot'] = df_1['label_seq_id'].apply(int) - df_1['pos_diff']
    df_1['resSeq_mcsa'] = df_1['label_seq_id'].apply(int)
    # df_1['label_seq_id']
    # df_1.iloc[166]
    # df_1[['resSeq_uniprot','auth_comp_id_code']].to_numpy()



    df_dbref_1['seq_len'] = df_dbref_1['seq_align_end'].apply(int)-df_dbref_1['seq_align_beg'].apply(int)
    df_dbref_1 = df_dbref_1.sort_values('seq_len',ascending=False).reset_index(drop=True)
    uniprot_accession_id = df_dbref_1['pdbx_db_accession'].to_list()[0]
    
    if len(df_uniprot_sites[df_uniprot_sites['uniprot_entry'].apply(lambda x:uniprot_accession_id in x)].reset_index(drop=True)) != 0:

        df_uniprot_sites = df_uniprot_sites[df_uniprot_sites['uniprot_entry'].apply(lambda x:uniprot_accession_id in x)].reset_index(drop=True)
        df_uniprot_sites['ec_1_len'] = df_uniprot_sites['ec'].apply(lambda x:len([h for h in x if h[:len(p1_ec)]==p1_ec]))
        df_uniprot_sites = df_uniprot_sites.sort_values('ec_1_len',ascending=False).reset_index(drop=True)
    
    seq_uniprot = df_uniprot_sites['final_record'].apply(lambda x:x.sequence)[0]

    seq_uniprot = seq_uniprot.replace('U','A').replace('O','A').replace('X','A').replace('B','A').replace('Z','A').replace('J','A')
    

    seq_cif = ''.join(df_1['auth_comp_id_code'].tolist())
    
    if 'pdb_pos_start' in df_uniprot_sites.columns:
        site_pos_st = int(df_uniprot_sites.iloc[0]['pdb_pos_start'])-1
        site_pos_ed = int(df_uniprot_sites.iloc[0]['pdb_pos_end'])-1
    else:
        site_pos_st = 0
        site_pos_ed = len(df_uniprot_sites.iloc[0]['final_record'].sequence)
        

    
    if len(seq_cif) > site_pos_ed-site_pos_st:
        seq_uniprot_1 = seq_uniprot[site_pos_st:]
    else:
        seq_uniprot_1 = seq_uniprot[site_pos_st:site_pos_ed]
    
    
    
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match = 5
    aligner.mismatch = -4
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    
    
    alignments = aligner.align(seq_uniprot_1, seq_cif)
    

    counter_seq = 0
    # len(seq_cif)
    seq_all = []
    for alignment in alignments:
        seq_all+=[[alignment,len(alignment.coordinates[0]),]]
        counter_seq += 1
        if counter_seq > 100:
            print('unaligned seq:',pdb_name)
            break
    
    seq_all = sorted(seq_all, key=lambda x: (x[1]))
    alignments_a = seq_all[0][0]
    

    align_seqA = alignments_a.coordinates[0]
    align_seqB = alignments_a.coordinates[1]
    # print(alignments_a)



    seqA=seq_uniprot[0:site_pos_st]
    seqB='-'*(site_pos_st)

    len(seq_uniprot)
    
    l_seq_uniprot = []
    for j in range(0, len(align_seqB)-1):
        if align_seqB[j] == align_seqB[j+1]:
            seqB += '-'*(align_seqA[j+1]-align_seqA[j])
        else:
            seqB += seq_cif[align_seqB[j]:align_seqB[j+1]]
            

        if align_seqA[j] == align_seqA[j+1]:
            seqA += '-'*(align_seqB[j+1]-align_seqB[j])
        else:
            seqA += seq_uniprot_1[align_seqA[j]:align_seqA[j+1]]

    if len(seqA)!=len(seqB):
        print('misaligned:',pdb_name)



            
            # l_seq_uniprot += [_ for _ in range(align_seqA[j],align_seqA[j+1])]
            
        
    l_seq_uniprot = []

    n = 1
    l_seq_uniprot = []
    
    for seqA_s,seqB_s in zip([x for x in seqA],[y for y in seqB]):
        
        if seqA_s == '-':
            l_seq_uniprot += [-1]
        elif seqB_s=='-':
            n+=1
            ''
        else:
            l_seq_uniprot += [n]
            
            n+=1
            
            
    n_neg = (np.array(l_seq_uniprot)<0).sum()*-1

    l_seq_uniprot_a = []
    for h in range(len(l_seq_uniprot)):
        
        if l_seq_uniprot[h] == -1 and l_seq_uniprot_a == []:
            l_seq_uniprot_a += [n_neg]
        elif l_seq_uniprot[h] == -1:
            l_seq_uniprot_a += [l_seq_uniprot_a[h-1]+1]
        else:
            l_seq_uniprot_a += [l_seq_uniprot[h]]


    df_1['resSeq_uniprot'] = l_seq_uniprot_a
    
    


    df_uniprot_sites_1 = df_uniprot_sites[df_uniprot_sites['pdb_id']==pdb_name_s].reset_index().drop('index',axis=1)
    
    df_uniprot_sites_1['pdb_chain_s'] = df_uniprot_sites_1['pdb_chain'].apply(lambda x:x.split('/'))
    
    


    def f_a(x):
        l_all = []
        for h in x:
            for q1 in range(h.location._start+1,h.location._end+1):
                l_all+=[int(q1)]
        return list(set(l_all))
            
    df_uniprot_sites_1['sites_pos'] = df_uniprot_sites_1['sites'].apply(lambda x:f_a(x))     

    df_uniprot_sites_1 = df_uniprot_sites_1.explode(['pdb_chain_s'])
    df_uniprot_sites_1 = df_uniprot_sites_1.explode(['sites_pos'])
    df_uniprot_sites_1['valid_chain_uniprot'] = True

    df_uniprot_sites_1 = df_uniprot_sites_1[df_uniprot_sites_1['sites_pos'].isnull()==False]
    
    # df_uniprot_sites_1['sites_pos'].astype(int)
    df_uniprot_sites_1 = df_uniprot_sites_1.drop_duplicates(subset=['pdb_id','pdb_chain_s','sites_pos'],keep='first')




    df_2 = df_1.merge(df_uniprot_sites_1[['pdb_chain_s','sites_pos','sites_metal','sites_active','sites_binding','sites_mutagen','valid_chain_uniprot']],how='left',left_on=['auth_asym_id','resSeq_uniprot'],right_on=['pdb_chain_s','sites_pos'])
    # df_uniprot_sites_1[['pdb_chain_s','sites_pos']].to_numpy()
    # df_1[['auth_asym_id','resSeq_uniprot']].to_numpy()

    # df_2[['pdb_chain_s','sites_pos']].to_numpy()

    df_uniprot_sites_data_1 = df_1.merge(df_uniprot_sites_1,left_on=['auth_asym_id','resSeq_uniprot'],right_on=['pdb_chain_s','sites_pos'])
    df_uniprot_sites_data_1['valid_chain'] = True
    
    
    
    df_mcsa_data_1 = df_mcsa_data[df_mcsa_data['PDB']==pdb_name_s].reset_index().drop('index',axis=1)


    df_mcsa_data_2 = df_mcsa_data_1.merge(df_1,how='inner',left_on=['chain/kegg compound','resid/chebi id'],right_on=['auth_asym_id','resSeq_mcsa'])
    df_mcsa_data_2['code'] = df_mcsa_data_2['code'].apply(lambda x:x.upper())
    # df_mcsa_all = df_mcsa_all.append(df_mcsa_data_2)
    
    if (df_mcsa_data_2['code'] != df_mcsa_data_2['label_comp_id']).sum()>0:
        print('wrong mcsa matching: ',pdb_name)

        df_mcsa_data_2['code_1l'] = df_mcsa_data_2['code'].apply(lambda x:codification[x.upper()])
    
    df_2 = df_2.merge(df_mcsa_data_1,how='left',left_on=['auth_asym_id','resSeq_mcsa'],right_on=['chain/kegg compound','resid/chebi id'])
    


    df_2['sites_combined'] = df_2.apply(f_valid_chain,axis=1)



    df_2.to_excel(path_output_img+pdb_name_s+'_weights.xlsx')

    
    
    
    
    


    # Draw



    for chain_id_s in set(df_1['auth_asym_id'].to_list()):
        df_1_s = df_1[df_1['auth_asym_id'] == chain_id_s].copy()


        if d_config.task_type in ['uniprot','parkinson','mcsa']:
            text = 'open '+pdb_name_s+' format mmcif \n'
        if d_config.task_name[:3]=='af_':
            text = 'alphafold fetch '+pdb_name_s.split('-')[0]+' format mmcif \n'

        text += 'set bgColor white\n'
        text += 'hide atoms\n'
        text += 'style stick\n'
        text += 'show /'+chain_id_s+' cartoons\n'



        
        text += "2dlab text 'Grad-CAM Weights (Top n%)' size 20 xpos .53 ypos .15\n"
        # text += "key #008bfb:0 #70c6a2:0.25 #fddb7f:0.5 #f46d45:0.75 #ff0051:1 pos .51,.09 size .32,.04\n"
        # text += "key #959595:0 #43dd18:0.25 #f9f938:0.5 #f03114:0.75 #e30000:1  pos .51,.12 size .32,.02\n"
        
        
        # text += "key #acacac: #FFE20A: #1C7C54: #F94824: #CC0000: pos .51,.12 size .32,.02 colorTreatment distinct numericLabelSpacing proportional \n"
        # text += "2dlab text '0' size 18 x .505 y .08; 2dlab text '0.2' size 18 x .56 y .08;2dlab text '0.4' size 18 x .625 y .08;2dlab text '0.6' size 18 x .69 y .08;;2dlab text '0.8' size 18 x .755 y .08;2dlab text '1' size 18 x .825 y .08;\n"
        
        # text += "key  "+l_colors[0]+": "+l_colors[1]+":  "+l_colors[2]+": "+l_colors[3]+": pos .51,.12 size .32,.02 colorTreatment distinct  numericLabelSpacing  proportional \n"
        # text += "2dlab text '0' size 18 x .505 y .08; 2dlab text '0.25' size 18 x .57 y .08;2dlab text '0.5' size 18 x .655 y .08; 2dlab text '0.75' size 18 x .73 y .08;2dlab text '1' size 18 x .825 y .08;\n"
        
        text += "key  "+l_colors[2]+":  "+l_colors[1]+": "+l_colors[0]+": pos .51,.12 size .32,.02 colorTreatment distinct  numericLabelSpacing  proportional \n"
        # text += "2dlab text '15%' x .74 y .08   size 18;2dlab text '10%' size 18 x .655 y .08; 2dlab text '5%' size 18 x .58 y .08;\n"
        # text += "2dlab text '10%' size 18 x .6175 y .08; 2dlab text '5%' size 18 x .545 y .08;\n"
        text += "2dlab text '0%' size 18 x 0.50 y .08; 2dlab text '5%' size 18 x .61 y .08; 2dlab text '10%' size 18 x .71 y .08; ; 2dlab text '100%' size 18 x .81 y .08; 2dlab text '...' size 18 x .77 y .09;\n"

        

        for chain_id_s1  in chain_all:
            if chain_id_s1 != chain_id_s:
                text += 'delete /'+chain_id_s1+'\n'

        
        for weights_s in set(set(df_1_s['weights'])):
            # if weights_s != 0:
                # break
            text_aa = ''
            for index,row in df_1_s[df_1_s['weights'] == weights_s].iterrows():
                text_aa += '/'+row['auth_asym_id']+':'+str(row['auth_seq_id'])+' '
            text += 'color '+text_aa+' '+matplotlib.colors.to_hex(cmap(weights_s))+'\n'
            

        # for index,row in df_mcsa_data_1.iterrows():
        #     text += 'shape sphere radius 2 center /'+str(row['chain/kegg compound'])+':'+str(row['resid/chebi id'])+' color #1F45FC70\n'

        for index,row in df_mcsa_data_2.iterrows():
            
            text += 'show /'+str(row['auth_asym_id'])+':'+str(row['auth_seq_id'])+' atoms\n'
            # break
            if row['weights'] >= 0.5:
                text += 'shape sphere radius 2 center /'+str(row['auth_asym_id'])+':'+str(row['auth_seq_id'])+' color #F8CBAD60\n'
            else:
                text += 'shape sphere radius 2 center /'+str(row['auth_asym_id'])+':'+str(row['auth_seq_id'])+' color #d7e5f860\n'

            # text += 'shape sphere radius 2 center /'+str(row['chain/kegg compound'])+':'+str(row['resid/chebi id'])+' color #0000ff50\n'



        with open(path_output_img+pdb_name_s+'_'+chain_id_s+'_mcsa.cxc','w') as f1:
            f1.writelines(text)





    for chain_id_s in set(df_1['auth_asym_id'].to_list()):
        df_1_s = df_1[df_1['auth_asym_id'] == chain_id_s].copy()


        if d_config.task_type in ['uniprot','parkinson']:
            text = 'open '+pdb_name_s+'\n'
        if d_config.task_name[:3]=='af_':
            text = 'alphafold fetch '+pdb_name_s.split('-')[0]+'\n'

        text += 'set bgColor white\n'
        text += 'hide atoms\n'
        
        text += "2dlab text 'Grad-CAM Weights (Top n%)' size 20 xpos .53 ypos .15\n"
        # text += "key #008bfb:0 #70c6a2:0.25 #fddb7f:0.5 #f46d45:0.75 #ff0051:1 pos .51,.09 size .32,.04\n"
        # text += "key #959595:0 #43dd18:0.25 #f9f938:0.5 #f03114:0.75 #e30000:1  pos .51,.12 size .32,.02\n"
        
        
        # text += "key #acacac: #FFE20A: #1C7C54: #F94824: #CC0000: pos .51,.12 size .32,.02 colorTreatment distinct numericLabelSpacing proportional \n"
        # text += "2dlab text '0' size 18 x .505 y .08; 2dlab text '0.2' size 18 x .56 y .08;2dlab text '0.4' size 18 x .625 y .08;2dlab text '0.6' size 18 x .69 y .08;;2dlab text '0.8' size 18 x .755 y .08;2dlab text '1' size 18 x .825 y .08;\n"
        
        # text += "key  "+l_colors[0]+": "+l_colors[1]+":  "+l_colors[2]+": "+l_colors[3]+": pos .51,.12 size .32,.02 colorTreatment distinct  numericLabelSpacing  proportional \n"
        # text += "2dlab text '0' size 18 x .505 y .08; 2dlab text '0.25' size 18 x .57 y .08;2dlab text '0.5' size 18 x .655 y .08; 2dlab text '0.75' size 18 x .73 y .08;2dlab text '1' size 18 x .825 y .08;\n"
        
        text += "key  "+l_colors[2]+":  "+l_colors[1]+": "+l_colors[0]+": pos .51,.12 size .32,.02 colorTreatment distinct  numericLabelSpacing  proportional \n"
        # text += "2dlab text '15%' x .74 y .08   size 18;2dlab text '10%' size 18 x .655 y .08; 2dlab text '5%' size 18 x .58 y .08;\n"
        # text += "2dlab text '10%' size 18 x .6175 y .08; 2dlab text '5%' size 18 x .545 y .08;\n"
        text += "2dlab text '0%' size 18 x 0.50 y .08; 2dlab text '5%' size 18 x .61 y .08; 2dlab text '10%' size 18 x .71 y .08; ; 2dlab text '100%' size 18 x .81 y .08; 2dlab text '...' size 18 x .77 y .09;\n"



        for chain_id_s1  in chain_all:
            if chain_id_s1 != chain_id_s:
                text += 'delete /'+chain_id_s1+'\n'

        
        for weights_s in set(set(df_1_s['weights'])):
            # if weights_s != 0:
                # break
            text_aa = ''
            for index,row in df_1_s[df_1_s['weights'] == weights_s].iterrows():
                text_aa += '/'+row['auth_asym_id']+':'+str(row['auth_seq_id'])+' '
            text += 'color '+text_aa+' '+matplotlib.colors.to_hex(cmap(weights_s))+'\n'


        # for index,row in df_mcsa_data_1.iterrows():
        #     text += 'shape sphere radius 2 center /'+str(row['chain/kegg compound'])+':'+str(row['resid/chebi id'])+' color #1F45FC70\n'
        df_uniprot_sites_data_1_s = df_uniprot_sites_data_1[df_uniprot_sites_data_1['auth_asym_id']==chain_id_s].copy()
        for index,row in df_uniprot_sites_data_1_s.iterrows():
            text += 'show /'+str(row['auth_asym_id'])+':'+str(row['auth_seq_id'])+' atoms\n'
            # break
            if row['weights'] >= 0.5:
                text += 'shape sphere radius 2 center /'+str(row['auth_asym_id'])+':'+str(row['auth_seq_id'])+' color #F8CBAD60\n'
            else:
                text += 'shape sphere radius 2 center /'+str(row['auth_asym_id'])+':'+str(row['auth_seq_id'])+' color #d7e5f860\n'

            # text += 'shape sphere radius 2 center /'+str(row['chain/kegg compound'])+':'+str(row['resid/chebi id'])+' color #0000ff50\n'



        with open(path_output_img+pdb_name_s+'_'+chain_id_s+'_uniprot_sites.cxc','w') as f1:
            f1.writelines(text)
    
    
    

    for chain_id_s in set(df_1['auth_asym_id'].to_list()):
        # break

        df_1_s = df_2[df_2['auth_asym_id']==chain_id_s].copy()
        
        # df_1_s['n'] = range(0, len(df_1_s))
        df_1_s['n'] = df_1_s['resSeq_uniprot'].apply(f_1)
        if df_1_s.iloc[0]['resSeq_uniprot']%10 < 9:
            df_1_s.at[0,'n'] = str(df_1_s.iloc[0]['resSeq_uniprot'])
        # df_1_s.at[len(df_1_s)-1,'n'] = str(df_1_s.iloc[-1]['resSeq_uniprot'])
        
        df_1_s_1 = df_1_s
        # df_1_s_1['mcsa'] = df_1_s_1['valid_chain'].apply(f_valid_chain)
        # df_1_s_1['final_label'] = df_1_s_1.apply(f_join,axis=1)
        
        weights_a1 = df_1_s['weights']



        amino_acid_len_s_st_min = df_1_s_1.iloc[0]['resSeq_uniprot']
        amino_acid_len_s_st_max = df_1_s_1.iloc[-1]['resSeq_uniprot']

        amino_acid_len_s_0 = int(np.floor(amino_acid_len_s_st_min/100)) 
        
        amino_acid_len =  int(np.ceil((amino_acid_len_s_st_max)/100))-int(np.floor(amino_acid_len_s_st_min/100))
        
        
        fig, ax1 = plt.subplots(nrows=amino_acid_len, figsize=(12, amino_acid_len))

        for amino_acid_len_s in range(int(np.floor(amino_acid_len_s_st_min/100)),int(np.ceil((amino_acid_len_s_st_max)/100))):
        
            amino_acid_len_s_st = amino_acid_len_s*100
            amino_acid_len_s_ed = (amino_acid_len_s+1)*100

            amino_acid_len_ax = amino_acid_len_s - amino_acid_len_s_0
            if amino_acid_len == 1:
                ax = ax1
            else:
                ax = ax1[amino_acid_len_ax]
            
            df_1_s_2 = df_1_s_1[(df_1_s_1['resSeq_uniprot']>amino_acid_len_s_st) & (df_1_s_1['resSeq_uniprot']<=amino_acid_len_s_ed)]
            
            df_ref = pd.DataFrame([range(amino_acid_len_s_st+1,amino_acid_len_s_ed+1)]).T
            
            # df_ref = pd.DataFrame([range(max(amino_acid_len_s_st_min,amino_acid_len_s_st+1)
            #                             ,min(amino_acid_len_s_st_max,amino_acid_len_s_ed+1))]).T
            
            df_ref = df_ref.rename({0:'resSeq_ref'},axis=1)
            df_1_s_3 = df_ref.merge(df_1_s_2,how='left',left_on='resSeq_ref',right_on='resSeq_uniprot')
            df_1_s_3['weights'] = df_1_s_3['weights'].fillna(-1)
            # weights_s = [cmap(h) for h in df_1_s_2['weights'].to_numpy()]
            weights_s = df_1_s_3['weights'].to_numpy()
            
            # extent = [df_1_s_2.iloc[0]['resSeq_uniprot']-amino_acid_len_s_st-1, df_1_s_2.iloc[-1]['resSeq_uniprot']-amino_acid_len_s_st,0,0.6]

            extent = [0,100,0,0.6]

            # plt.figure(figsize=(10,0.5))

            # plt.bar(range(len(df_1_s_2)),weights_s,color=cmap(weights_s),width=1)
            # plt.ylim([0,1])

            ax.imshow(weights_s[np.newaxis,:]+0.0001,cmap=cmap,norm=matplotlib.colors.Normalize(vmin=0, vmax=1), aspect="auto", extent=extent)
            

            ax.set_yticks([])
            

            
            # x_label = df_1_s_2['n'].to_list()
            x_mcsa = df_1_s_3['sites_combined'].to_list()

            for h1 in range(len(x_mcsa)):
                # if len(x_label[h1]) > 1:
                if x_mcsa[h1] in ['M','B']:
                    # print(h1)
                    ax.add_patch(matplotlib.patches.Rectangle((h1,0), 1, extent[3], hatch='///', fill=False, snap=False))
                if DRAW_UNIPROT_SITES == 1:
                    if x_mcsa[h1] in ['U','B']:
                        # print(h1)
                        ax.add_patch(matplotlib.patches.Rectangle((h1,0), 1, extent[3], hatch='\\\\\\', fill=False, snap=False))


            # ax[amino_acid_len_s].set_xticklabels(x_label,fontsize=14)
            # ax[amino_acid_len_s].set_xticks([a+0.5 for a in range(len(weights_s))])
    
            x_label_major = df_1_s_2[df_1_s_2['n']!='']['n'].to_list()
            x_label_minor = df_1_s_2['resSeq_uniprot'].tolist()

            
            
            
            x_label_major = df_1_s_2[df_1_s_2['n']!='']['n'].to_list()
            x_label_minor = df_1_s_2['resSeq_uniprot'].tolist()

            ax.set_xticks([(int(a)-amino_acid_len_s_st)/10*10-0.5 for a in x_label_major])
            ax.set_xticks([a-amino_acid_len_s_st-0.5 for a in x_label_minor], minor = True)
            ax.set_xticklabels(x_label_major,fontsize=14)
            # x_label_s = []
            # for x_label_n in zip(x_label,range(len(x_label))):

            #     if x_label_n[1]%10 == 0:
            #         x_label_s += [x_label_n[0]+'\n'+str(amino_acid_len_s_st+x_label_n[1])]
            #     else:
            #         x_label_s += [x_label_n[0]]


            ax.tick_params(which='both', width=1.5)
            ax.tick_params(which='major', length=6)
            ax.tick_params(which='minor', length=4)
            
            ax.set_xlim(0,100)
            

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        fig.tight_layout()
        plt.savefig(path_output_img+pdb_name_s+'_'+chain_id_s+'.png',dpi=300)
        plt.close()

    

def download_uniprot_record(uniprot_id_s):
    
    os.makedirs('uniprotkb',exist_ok=True)
    
    
    l_uniprot_sprot = []
    
    for uniprot_id_s1 in uniprot_id_s:
        
        
        
        if os.path.exists('uniprotkb/'+uniprot_id_s1+'.txt')==False:
            filename = wget.download('https://rest.uniprot.org/uniprotkb/'+uniprot_id_s1+'.txt', out='uniprotkb/'+uniprot_id_s1+'.txt')
        
        
        with open('uniprotkb/'+uniprot_id_s1+'.txt') as handle:
            record = SwissProt.read(handle)
        
            l_uniprot_sprot += [[record.entry_name,record.accessions,record]]
    
    
    df_all_1 = pd.DataFrame(l_uniprot_sprot,columns=['uniprot_entry_name','uniprot_entry','final_record'])
            
    df_all_1['PDB_record'] = df_all_1['final_record'].apply(lambda x:[s1 for s1 in x.cross_references if s1[0] == 'PDB'])

    df_all_1['AF_record'] = df_all_1['final_record'].apply(lambda x:[s1 for s1 in x.cross_references if s1[0] == 'AlphaFoldDB'])

        
    df_all_1['pdb_id'] = df_all_1['AF_record'].apply(lambda x:x[0][1].lower())+'-f1'
    df_all_1['pdb_chain'] = 'A'
    
    
    df_all_1['desc'] = df_all_1['final_record'].apply(lambda x:[h for h in x.description.split('; ') if h.find('EC=')>=0])
    df_all_1['desc'] = df_all_1['desc'].apply(lambda x:[h.replace(';','') for h in x])
    
    
    
    df_all_1['ec'] = df_all_1['desc'].apply(lambda x:[h.split('=')[1].split(' ')[0] for h in x])
    df_all_1[df_all_1['desc'].apply(lambda x:len([h for h in x if h.find('PubMed')>=0]))>0]

    df_all_1['sites_metal'] = df_all_1['final_record'].apply(lambda x:[s1 for s1 in x.features if s1.type in ['METAL']])
    df_all_1['sites_active'] = df_all_1['final_record'].apply(lambda x:[s1 for s1 in x.features if s1.type in ['ACT_SITE']])
    df_all_1['sites_binding'] = df_all_1['final_record'].apply(lambda x:[s1 for s1 in x.features if s1.type in ['BINDING']])
    df_all_1['sites_mutagen'] = df_all_1['final_record'].apply(lambda x:[(s1,s1.qualifiers) for s1 in x.features if s1.type in ['MUTAGEN']])

    df_all_1['sites'] = df_all_1['sites_metal']+df_all_1['sites_active'] +df_all_1['sites_binding']

    df_all_1['enzyme_group'] = df_all_1['ec']
    
    return df_all_1
    





if __name__ == "__main__":

    datetime_st = datetime.datetime.now()


    parser = ArgumentParser()
    parser.add_argument("-s", "--task_name", dest="task_name",default='example')
    parser.add_argument("-t", "--target_type", dest="target_type",default='cif',help='cif or af2')
    

    if '-f' in sys.argv or '--ip=127.0.0.1' in sys.argv:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
      
    
    args.target_source = './data/'+args.task_name
    args.distance = '9'
    


    test_loader = load_pdb_data(args,1)
    


    


    datetime_s1 = datetime.datetime.now()
    d_model_datetime = {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = torch.load('./models/model.pt', map_location=device)
    model.eval()

    
    with open('./excel/uniprot_sites.pkl','rb') as f1:
        df_uniprot_sites = pickle.load(f1)

    df_mcsa_data = pd.read_excel('./excel/mcsa_data_t.xlsx').drop('Unnamed: 0',axis=1)
    
    df_ec = pd.read_excel('./ec.xlsx')
    l_ec = df_ec['ec'].tolist()
    l_thres = df_ec['d_eval_test.thresholds'].tolist()
    
    l_method = [
                'Saliency',
                ]
    
    
    
    
    d_result = {}

    for local_batch in test_loader:
        # break
        
        
        
        for uniprot_id_s in local_batch.uniprot_id:
            
            df_uniprot_sites_1 = df_uniprot_sites[df_uniprot_sites['uniprot_entry'].apply(lambda x:uniprot_id_s[0] in x)].reset_index(drop=True)
            
            
        if len(df_uniprot_sites_1)==0:
                
            uniprot_id_s = local_batch.uniprot_id[0]
            
            
            df_uniprot_sites_1 = download_uniprot_record(uniprot_id_s)
        
        
        
        
            
        local_batch=local_batch.to(device)

        out = model(local_batch.x, local_batch.edge_index, local_batch.batch)
        # pred = torch.sigmoid(out).squeeze(dim=1)

        # pred_proba = torch.sigmoid(model1(local_batch.to(device)))

        pred = torch.sigmoid(out).cpu()
        l1 = pred >= torch.Tensor(l_thres)

        indices = l1.nonzero()

        d_1 = {}



        # local_batch.requires_grad = True
        for indices_s in indices:
            # break
            

            predicted_indices = indices_s[1]
            

            



            # d_1[l_ec[predicted_indices]] = {}
            d_1[l_ec[predicted_indices]] = float(pred[:,predicted_indices].cpu().detach())




            
            if len(l_ec[predicted_indices].split('.-')[0].split('.'))>=3:
                    
                task_level = 'graph'
                model_config = ModelConfig(
                    mode='multiclass_classification',
                    task_level=task_level,
                    return_type='raw',
                )


                explainer = Explainer(
                    model,
                    algorithm=CaptumExplainer('Saliency'),
                    explanation_type='phenomenon',
                    edge_mask_type='object',
                    node_mask_type='attributes',
                    model_config=model_config,
                )



                explanation = explainer(
                    local_batch.x,
                    local_batch.edge_index,
                    # index=1,
                    target=predicted_indices,
                    batch=local_batch.batch,
                )








                node_mask_result = explanation.node_mask.mean(axis=1).cpu().detach().numpy()

                





                pdb_name = local_batch.name[0]
                
                d_config = args
                d_config.task_type = 'uniprot'
                d_config.path_pdb = './data/'+args.task_name+'/'
                d_config.path_output_s1 = './results/'+args.task_name+'/'
                d_config.ec = l_ec[predicted_indices]
                d_config.pdb_name = pdb_name
                d_config.heatmap_method = 'torch_captum Saliency'
                
                df_mcsa = pd.DataFrame()
                
                v = node_mask_result
                
                
                k = pdb_name
                draw_prediction(k,v,d_config,df_uniprot_sites_1,df_mcsa_data)
                
            
            
            
        d_result[pdb_name] = d_1



    df_results = pd.DataFrame(d_result).T
    df_results = df_results[df_results.columns.sort_values()]
    df_results = df_results.fillna('')
    
    os.makedirs('./results/'+args.task_name,exist_ok=True)
    df_results.to_excel('./results/'+args.task_name+'/prediction.xlsx')

        