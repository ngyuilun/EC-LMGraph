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
import numpy as np
from sklearn import preprocessing
import datetime
os.umask(0)
from sklearn.preprocessing import MinMaxScaler
from utils.model_gcn_mo import *
from draw_figures import *

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


    model = torch.load('./models/model.pt', map_location=device, weights_only=False)
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

        