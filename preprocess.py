import os
import multiprocessing
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import torch
from torch_geometric.data import Data
import gemmi

from utils.load_excel_pkl import load_excel_pkl

def load_mmcif_data(data_list, args, pdb_id_chain, pdb_name, pdb_name_chain):
    """
    Reads a .cif file, extracts specific chain atoms, computes distances,
    and constructs the graph nodes and edges.
    """
    codification = {
        "ALA": 'A', "CYS": 'C', "ASP": 'D', "GLU": 'E', "PHE": 'F',
        "GLY": 'G', "HIS": 'H', "ILE": 'I', "LYS": 'K', "LEU": 'L',
        "MET": 'M', "ASN": 'N', "PYL": 'O', "PRO": 'P', "GLN": 'Q',
        "ARG": 'R', "SER": 'S', "THR": 'T', "SEC": 'U', "VAL": 'V',
        "TRP": 'W', "TYR": 'Y'
    }
    l_amino_acid = list(codification.keys())
    l_amino_acid_all = l_amino_acid + ['UNK']

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l_amino_acid_all)
    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

    # Read and parse CIF file using gemmi
    cif_file = os.path.join(args.path_raw_cif, f"{pdb_name}.cif")
    if not os.path.exists(cif_file):
        print(f"File not found: {cif_file}")
        return

    block = gemmi.cif.read_file(cif_file).sole_block()

    col_atom_site = ['group_PDB', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'auth_seq_id', 
                     'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'pdbx_PDB_model_num']

    df_atoms = pd.DataFrame(block.find('_atom_site.', col_atom_site), columns=col_atom_site)
    if df_atoms.empty:
        return
        
    model_num = df_atoms['pdbx_PDB_model_num'].iloc[0]

    # Filter for valid atoms and the specific chain
    df_atoms = df_atoms[(df_atoms['group_PDB'] == 'ATOM') & 
                        (df_atoms['pdbx_PDB_model_num'] == model_num) & 
                        (df_atoms['auth_asym_id'] == pdb_name_chain)].reset_index(drop=True)
    
    df_atoms['auth_comp_id'] = df_atoms['auth_comp_id'].apply(lambda x: 'UNK' if x not in l_amino_acid else x)
    df_atoms['chainID_resSeq'] = df_atoms['auth_asym_id'] + df_atoms['auth_seq_id'].str.zfill(5)
    df_atoms = df_atoms.drop_duplicates(['chainID_resSeq', 'auth_atom_id']).reset_index(drop=True)
    
    df_atoms['Cartn_x'] = df_atoms['Cartn_x'].astype(float)
    df_atoms['Cartn_y'] = df_atoms['Cartn_y'].astype(float)
    df_atoms['Cartn_z'] = df_atoms['Cartn_z'].astype(float)

    # Prioritize CA, then CB, then CE atoms for coordinates
    for atom_type in ['CA', 'CB', 'CE']:
        df_atoms[f'Cartn_x_{atom_type}'] = df_atoms['Cartn_x']
        df_atoms[f'Cartn_y_{atom_type}'] = df_atoms['Cartn_y']
        df_atoms[f'Cartn_z_{atom_type}'] = df_atoms['Cartn_z']
        df_atoms[f'Cartn_n_{atom_type}'] = 1
        mask = df_atoms['auth_atom_id'] != atom_type
        df_atoms.loc[mask, [f'Cartn_x_{atom_type}', f'Cartn_y_{atom_type}', f'Cartn_z_{atom_type}', f'Cartn_n_{atom_type}']] = 0

    df_1 = df_atoms.groupby('chainID_resSeq').agg({
        'auth_comp_id': 'first', 'group_PDB': 'count',
        'Cartn_x_CA': 'sum', 'Cartn_y_CA': 'sum', 'Cartn_z_CA': 'sum', 'Cartn_n_CA': 'sum',
        'Cartn_x_CB': 'sum', 'Cartn_y_CB': 'sum', 'Cartn_z_CB': 'sum', 'Cartn_n_CB': 'sum',
        'Cartn_x_CE': 'sum', 'Cartn_y_CE': 'sum', 'Cartn_z_CE': 'sum'
    })

    df_1['group_PDB'] = df_1['group_PDB'].astype(float)
    df_1['Cartn_x_CE'] /= df_1['group_PDB']
    df_1['Cartn_y_CE'] /= df_1['group_PDB']
    df_1['Cartn_z_CE'] /= df_1['group_PDB']

    # Fallback coordinate logic
    df_1['Cartn_x'] = df_1['Cartn_x_CB']
    df_1['Cartn_y'] = df_1['Cartn_y_CB']
    df_1['Cartn_z'] = df_1['Cartn_z_CB']

    mask_no_ca = df_1['Cartn_n_CA'] != 1
    df_1.loc[mask_no_ca, ['Cartn_x', 'Cartn_y', 'Cartn_z']] = df_1.loc[mask_no_ca, ['Cartn_x_CB', 'Cartn_y_CB', 'Cartn_z_CB']].values

    df_1['Cartn_n_CA_CB'] = df_1['Cartn_n_CA'] + df_1['Cartn_n_CB']
    mask_no_ca_cb = df_1['Cartn_n_CA_CB'] == 0
    df_1.loc[mask_no_ca_cb, ['Cartn_x', 'Cartn_y', 'Cartn_z']] = df_1.loc[mask_no_ca_cb, ['Cartn_x_CE', 'Cartn_y_CE', 'Cartn_z_CE']].values

    # Distance Matrix & Edge Generation
    coords = df_1[['Cartn_x', 'Cartn_y', 'Cartn_z']].values
    d1 = distance_matrix(coords, coords)

    if args.weighted_edge:
        d1[d1 > args.distance] = 0
        d2 = 1 - (d1 / args.distance)
        d2[d2 == 1] = 0
    else:
        d2 = ((d1 > 0) & (d1 < args.distance)).astype(int)
        
    coo_edges = coo_matrix(d2)
    
    # Node features (One-Hot Encoded Amino Acids)
    node_labels = df_1['auth_comp_id'].values
    node_labels_1 = onehot_encoder.transform(label_encoder.transform(node_labels).reshape(-1, 1))
    
    data = [pdb_id_chain, node_labels_1, coo_edges]
    data_list.append(data)

def process_dataset(args):
    """
    Main orchestration function. Reads the target list and processes CIFs in parallel.
    """
    df_enzyme_pdb = load_excel_pkl(args.path_excel_target)
    df_enzyme_pdb['pdb_id'] = df_enzyme_pdb['pdb_id'].astype(str).str.lower()
    
    target_pdb_list = df_enzyme_pdb[['pdb_id_chain', 'pdb_id', 'pdb_chain']].to_numpy()
    
    chunk_size = 10000
    n_dataset = int(np.ceil(len(df_enzyme_pdb) / chunk_size))

    os.makedirs(args.path_processed_graphs, exist_ok=True)

    for n_dataset_s in range(n_dataset):
        dfs_list = multiprocessing.Manager().list()
        tasks = []
        
        batch_targets = target_pdb_list[n_dataset_s * chunk_size : (n_dataset_s + 1) * chunk_size]
        
        for target in batch_targets:
            tasks.append((dfs_list, args, target[0], target[1], target[2]))
            
        print(f"Processing chunk {n_dataset_s+1}/{n_dataset}. Tasks: {len(tasks)}")
        
        ncpu = min(multiprocessing.cpu_count() - 1, 60)
        with multiprocessing.Pool(processes=ncpu) as pool:
            pool.starmap(load_mmcif_data, tasks)

        data_list = []
        for l_data_s in dfs_list:
            pdb_id_chain, node_labels_1, coo_edges = l_data_s
            
            x = torch.from_numpy(node_labels_1).type(torch.FloatTensor)
            y = torch.tensor([[0]], dtype=torch.float64) # Placeholder target
            edge_index = torch.tensor(np.array([coo_edges.row, coo_edges.col]), dtype=torch.int64)
            edge_weight = torch.tensor(np.array(coo_edges.data), dtype=torch.float32)
            
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight, name=pdb_id_chain)
            data_list.append(data)

        out_path = os.path.join(args.path_processed_graphs, f'data_{n_dataset_s}.pt')
        torch.save(data_list, out_path)
        print(f"Saved {len(data_list)} graphs to {out_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess raw CIF files into PyTorch Geometric graphs.")
    parser.add_argument("--path_excel_target", type=str, required=True, help="Path to the target Excel/pkl file.")
    parser.add_argument("--path_raw_cif", type=str, required=True, help="Path to the directory containing raw .cif files.")
    parser.add_argument("--path_processed_graphs", type=str, default="./processed_graphs", help="Where to save the .pt files.")
    parser.add_argument("--distance", type=int, default=9, help="Distance threshold for edge creation.")
    parser.add_argument("--weighted_edge", action='store_true', help="Use weighted edges instead of binary.")
    
    args = parser.parse_args()
    process_dataset(args)