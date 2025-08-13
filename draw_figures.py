

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
import math
import os
from Bio.Align import PairwiseAligner


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

    
if __name__ == '__main__':
    ''