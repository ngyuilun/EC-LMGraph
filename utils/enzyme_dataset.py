import os
import math
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, Data
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
import esm

# Adjust import assuming this is running from the root directory or inside utils
from utils.load_excel_pkl import load_excel_pkl

class cif_lm_dataset(InMemoryDataset):
    def __init__(self, root, path_pdb_target_dt_pkl, p_xlsx_target_task, transform=None, pre_transform=None):
        self.p_xlsx_target_task = p_xlsx_target_task
        self.path_pdb_target_dt_pkl = path_pdb_target_dt_pkl
        self.df_enzyme_pdb = load_excel_pkl(self.p_xlsx_target_task)
        
        super().__init__(root, transform, pre_transform)
        
        data_list = []
        for idx in self.processed_file_names:
            data_list += torch.load(os.path.join(self.processed_dir, idx))
            print("data loaded:", idx)

        self.data, self.slices = self.collate(data_list)
        data_list = None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_'+str(h)+'.pt' for h in range(math.ceil(len(self.df_enzyme_pdb)/10000))]

    def download(self):
        pass

    def process(self):
        n_dataset = math.ceil(len(self.df_enzyme_pdb)/10000)
        self.df_enzyme_pdb['pdb_id'] = self.df_enzyme_pdb['pdb_id'].apply(lambda x: x.lower())
        df_enzyme_pdb_s_1 = self.df_enzyme_pdb[['pdb_file_name','pdb_id','pdb_chain']]
        target_pdb_list = df_enzyme_pdb_s_1['pdb_file_name'].to_numpy()

        transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_lm = T5EncoderModel.from_pretrained(transformer_link)
        if 'cuda' in str(device):
            model_lm.half()  # Half-precision for GPU to save VRAM
        model_lm = model_lm.to(device)
        model_lm.eval()
        tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)

        batch_size = 1  
        n = 0
        for n_dataset_s in range(n_dataset):
            data_list = []
            for i in tqdm(range(0, len(target_pdb_list[n_dataset_s*10000:(n_dataset_s+1)*10000]), batch_size)):
                batch_pdb = target_pdb_list[n_dataset_s*10000:(n_dataset_s+1)*10000][i:i + batch_size]
                
                batch_sequences = []
                batch_pdbs = []
                for target_pdb_list_s in batch_pdb:
                    with open(self.path_pdb_target_dt_pkl + '/' + target_pdb_list_s + '.pkl', 'rb') as f1:
                        pdb_target = pickle.load(f1)
                    sequence = pdb_target['node_auth_comp_id_code']
                    sequence_spaced = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                    batch_sequences.append(sequence_spaced)
                    batch_pdbs.append(pdb_target)

                ids = tokenizer.batch_encode_plus(batch_sequences, add_special_tokens=True, padding="longest")
                input_ids = torch.tensor(ids['input_ids']).to(device)
                attention_mask = torch.tensor(ids['attention_mask']).to(device)

                with torch.no_grad():
                    embedding_repr = model_lm(input_ids=input_ids, attention_mask=attention_mask)

                for j, pdb_target in enumerate(batch_pdbs):
                    emb_0 = embedding_repr.last_hidden_state[j, :len(pdb_target['node_auth_comp_id_code'])].cpu() 
                    x1 = torch.from_numpy(pdb_target['node_attributes']).type(torch.FloatTensor).cpu() 
                    x = torch.cat((x1, emb_0), 1)

                    y = torch.tensor([[0]], dtype=torch.float64)
                    coo_edges = pdb_target['edge_attributes'].tocoo()
                    edge_index = torch.tensor(np.array([coo_edges.row, coo_edges.col]), dtype=torch.int64)
                    edge_weight = torch.tensor(np.array(coo_edges.data), dtype=torch.float32)

                    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight, name=batch_pdb[j])
                    data_list.append(data)

                del input_ids, attention_mask, embedding_repr, ids
                torch.cuda.empty_cache() 

            torch.save(data_list, os.path.join(self.processed_dir, 'data_{}.pt'.format(n)))
            n += 1

        del tokenizer, model_lm
        torch.cuda.empty_cache() 


class cif_esm_dataset(InMemoryDataset):
    def __init__(self, root, path_pdb_target_dt_pkl, p_xlsx_target_task, transform=None, pre_transform=None):
        self.p_xlsx_target_task = p_xlsx_target_task
        self.path_pdb_target_dt_pkl = path_pdb_target_dt_pkl
        self.df_enzyme_pdb = load_excel_pkl(self.p_xlsx_target_task)
        
        super().__init__(root, transform, pre_transform)
        
        data_list = []
        for idx in self.processed_file_names:
            data_list += torch.load(os.path.join(self.processed_dir, idx))
            print("data loaded:", idx)

        self.data, self.slices = self.collate(data_list)
        data_list = None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_'+str(h)+'.pt' for h in range(math.ceil(len(self.df_enzyme_pdb)/10000))]

    def download(self):
        pass

    def process(self):
        n_dataset = math.ceil(len(self.df_enzyme_pdb)/10000)
        self.df_enzyme_pdb['pdb_id'] = self.df_enzyme_pdb['pdb_id'].apply(lambda x: x.lower())
        df_enzyme_pdb_s_1 = self.df_enzyme_pdb[['pdb_file_name','pdb_id','pdb_chain']]
        target_pdb_list = df_enzyme_pdb_s_1['pdb_file_name'].to_numpy()

        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  

        n = 0
        for n_dataset_s in range(n_dataset):
            data_list = []
            for target_pdb_list_s in tqdm(target_pdb_list[n_dataset_s*10000:(n_dataset_s+1)*10000]):
                
                with open(self.path_pdb_target_dt_pkl+'/'+target_pdb_list_s+'.pkl','rb') as f1:
                    pdb_target = pickle.load(f1)
                        
                sequence_examples = [pdb_target['node_auth_comp_id_code']]
                sequence_examples = ["".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
                        
                data = [("protein1", sequence_examples[0])]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]

                emb_0 = token_representations[0][1:-1]
                x1 = torch.from_numpy(pdb_target['node_attributes']).type(torch.FloatTensor)
                x = torch.cat((x1, emb_0.cpu()), 1)

                y = torch.tensor([[0]], dtype=torch.float64)
                coo_edges = pdb_target['edge_attributes'].tocoo()
                edge_index = torch.tensor(np.array([coo_edges.row, coo_edges.col]), dtype=torch.int64)
                edge_weight = torch.tensor(np.array(coo_edges.data), dtype=torch.float32)

                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight, name=target_pdb_list_s)
                data_list.append(data)
            
            torch.save(data_list, os.path.join(self.processed_dir, 'data_{}.pt'.format(n)))
            n += 1
        
        del model


class cif_protbert_dataset(InMemoryDataset):
    def __init__(self, root, path_pdb_target_dt_pkl, p_xlsx_target_task, transform=None, pre_transform=None):
        self.p_xlsx_target_task = p_xlsx_target_task
        self.path_pdb_target_dt_pkl = path_pdb_target_dt_pkl
        self.df_enzyme_pdb = load_excel_pkl(self.p_xlsx_target_task)
        
        super().__init__(root, transform, pre_transform)
        
        data_list = []
        for idx in self.processed_file_names:
            data_list += torch.load(os.path.join(self.processed_dir, idx))
            print("data loaded:", idx)

        self.data, self.slices = self.collate(data_list)
        data_list = None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_'+str(h)+'.pt' for h in range(math.ceil(len(self.df_enzyme_pdb)/10000))]

    def download(self):
        pass
    
    def process(self):
        n_dataset = math.ceil(len(self.df_enzyme_pdb)/10000)
        self.df_enzyme_pdb['pdb_id'] = self.df_enzyme_pdb['pdb_id'].apply(lambda x: x.lower())
        df_enzyme_pdb_s_1 = self.df_enzyme_pdb[['pdb_file_name','pdb_id','pdb_chain']]
        target_pdb_list = df_enzyme_pdb_s_1['pdb_file_name'].to_numpy()

        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = AutoModel.from_pretrained("Rostlab/prot_bert")
        model.eval()  

        n = 0
        for n_dataset_s in range(n_dataset):
            data_list = []
            for target_pdb_list_s in tqdm(target_pdb_list[n_dataset_s*10000:(n_dataset_s+1)*10000]):
                
                with open(self.path_pdb_target_dt_pkl+'/'+target_pdb_list_s+'.pkl','rb') as f1:
                    pdb_target = pickle.load(f1)
                        
                sequence_examples = [pdb_target['node_auth_comp_id_code']]
                sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

                inputs = tokenizer(sequence_examples, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                
                token_representations = outputs.last_hidden_state
                emb_0 = token_representations[0][1:-1]
                x1 = torch.from_numpy(pdb_target['node_attributes']).type(torch.FloatTensor)
                x = torch.cat((x1, emb_0.cpu()), 1)

                y = torch.tensor([[0]], dtype=torch.float64)
                coo_edges = pdb_target['edge_attributes'].tocoo()
                edge_index = torch.tensor(np.array([coo_edges.row, coo_edges.col]), dtype=torch.int64)
                edge_weight = torch.tensor(np.array(coo_edges.data), dtype=torch.float32)

                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight, name=target_pdb_list_s)
                data_list.append(data)
            
            torch.save(data_list, os.path.join(self.processed_dir, 'data_{}.pt'.format(n)))
            n += 1

        del tokenizer, model