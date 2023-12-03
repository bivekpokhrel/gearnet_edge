


    

import torch
from torchdrug import models, transforms, datasets, data, layers, tasks, core
from torchdrug.layers import geometry
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils import data as torch_data
import pandas as pd
import os
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler




truncuate_transform = transforms.TruncateProtein(max_length=10, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([truncuate_transform, protein_view_transform])
# transform = transforms.ProteinView(view='residue')

# p_dataset = EnzymeCommissionToy("~/protein-datasets/", transform=transform, atom_feature='position', bond_feature=None)
# train_set, p_valid_set, p_test_set =p_dataset.split()
l1 = ['5xes', '1qcm', '2nb7', '2mj1', '1s4w', '2luv', '2lke', '1plp', '3j1r', '2nb8', '1ba4', '1amb', '2mj2', '7ck5']
l2 =['6zkc', '6zkk', '7zdh', '7b93', '7qsm', '7v2e', '7v31', '7v2c', '7v2r', '7v2h', '7v33', '5mdx', '7qsl', '7fdc', '7fda', '7fdb', '7qsk', '7t3q', '7vhp', '2ybb', '7oui', '7o01', '7t3t', '7t3r', '6vq8', '6vq6', '6vq7', '6k33', '7ard', '6mu1', '7dr2', '6mu2', '7vd5', '6dqj', '7wi3', '5zdh', '6dqn', '6dqz', '6dr0', '7dwx', '6dqs', '6dqv', '6dr2', '7unf', '6jlu', '6qwj', '7u8p', '7u8q', '6tcl', '6gy6', '7lhe', '7lhf', '7u8o', '7b9s', '7u4t', '7npr', '7np7', '7wg5', '7bin', '7rpm', '6hcg', '7f9o', '7dkf', '7bgl', '5xth', '7dgs', '7dgr', '3j8h', '7k0t', '7tdj', '6kig', '7tdk', '7tdg', '7tdi', '7tdh', '7pin', '7piw', '7cbl', '6kif', '7vmo', '7vmq', '7vmp', '7vms', '5go9', '5goa', '6rd4', '6tmk', '5gl1', '6ji8', '6yny', '5gl0', '5gkz', '5gky', '8cwm', '7t64', '7t65', '7m6l', '7m6a', '7tzc', '5xti']
class ProteinLoader(data.ProteinDataset):
    def __init__(self,csv_path,transform, random_seed=None ):
        # self.pdb_folder ='/ocean/projects/bio230029p/bpokhrel/data/alpha_no_trim'
        self.pdb_folder = '/ocean/projects/bio230029p/bpokhrel/data/pdb_m15_ac'
        # self.label_folder = '/ocean/projects/bio230029p/bpokhrel/data/new2_z_trans'
        self.input_counts=10000
        self.min_counts=100
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        
        self.csv_path = csv_path
        self.transform = transform
        self.label_dict, self.trans_df, self.k, self.label_population = self.make_dict()
        self.pdb_list = self.trans_df['pdbid'].tolist()
        self.pdbs = self.make_pdbs()
        
        self.data, self.pdb_list2, self.pdb_list3, self.pdb_name = self.load_proteins() # self.data needs to be there for the class
        self.label_list=self.get_label()
        self.targets = {'label': self.label_list} # needs to be there for the class
        self.num_samples = len(self.data)
        # self.num_classes = len(class_count)
       


        
    def make_dict(self):
        """1 only once needed to also crete folders gives dictionaty"""
        df = pd.read_csv(self.csv_path)

        trans_a_df = df[df['type_id'] == 1]  # Selecting only the transmembrane protien
        trans_b_df = trans_a_df.iloc[:self.input_counts, :].copy()  # make a copy of 50 pdb to avoid error
        trans_b_df['pdbid'] = trans_b_df['pdbid'].str.replace('[^\w]', '', regex=True)  # remove "=...." extra charecters
        # trans_b_df  = trans_b_df[~trans_b_df['pdbid'].isin(l1)]
        # trans_b_df  = trans_b_df[~trans_b_df['pdbid'].isin(l2)]
        counts = trans_b_df['membrane_name_cache'].value_counts()
        label_dict_counts = {key: counts[key] for key in counts.index}  # to make another dictionary to select labels
        print('label_dict_counts')
        print(label_dict_counts)
        total_counts = sum(label_dict_counts.values())
        
        selected_list = [key for key, value in label_dict_counts.items() if value > self.min_counts]
        # print(f'selected list : {selected_list}')
        # k=len(selected_list)
        trans_df = trans_b_df[trans_b_df['membrane_name_cache'].isin(selected_list)]
        location = trans_df['membrane_name_cache'].unique()
        label_dict = {key: value for value, key in enumerate(sorted(location))}
        class_names = list(label_dict.keys())
        print('label_dict')
        print(label_dict)
        print('class_names')
        print(class_names)
        print(f'total count {total_counts}')
        k = len(label_dict)
        label_population = {class_names[i]: label_dict_counts[class_names[i]] for i in range(k)}

        def select_100_rows(group):
            if len(group) >= 100:
                return group.sample(n=100)
            else:
                return group

        # Group by 'membrane_name_cache' and apply the function to select 100 rows from each group
        selected_df = trans_df.groupby('membrane_name_cache', group_keys=False).apply(select_100_rows)
        selected_df['original_index'] = selected_df['pdbid'].index
        counts2 = selected_df['membrane_name_cache'].value_counts()
        print(counts2)
        label_dict_counts2 = {key: counts2[key] for key in counts2.index}  # to make another dictionary to select labels
        print('label_dict_counts2')
        print(label_dict_counts2)
        total_counts2 = sum(label_dict_counts2.values())
        print(total_counts2)
        # Reset the index of the resulting DataFrame
        # selected_df.reset_index(drop=True, inplace=True)

        

        # Now, 'selected_df' contains only 100 rows from each 'membrane_name_cache'
        # print(selected_df)

        # return label_dict, trans_df, k, label_population
        return label_dict, selected_df, k, label_population #( selects 100 from each class)
    
    # def make_pdbs(self):
    #     file_names = os.listdir(self.label_folder)

    #     pdbs = [file for file in file_names if not file.startswith(".")]
    #     return pdbs

    def make_pdbs(self):
        # pdbs = self.trans_df['pdbid'] + '.pdb' # for other folder where pdb.pdb is there
        pdbs = self.trans_df['pdbid']
        return pdbs

    

    
    def load_proteins(self):
        protein_list=[]
        pdb_name = []
        pdb_list2=[] # protein none pdbs
        pdb_list3 = []
        for pdb in self.pdbs:  ## selecting pdb from the dataframe..
            try:
                
                pdb_path = os.path.join(self.pdb_folder, pdb)
                

                iprotein = data.Protein.from_pdb(pdb_path, atom_feature="position", bond_feature="length", residue_feature="symbol") #if atom feature = position input_dim =3 and when tranform=true it is 21
                filter_alpha = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])
                protein = filter_alpha(iprotein)
                if protein is None:

                    print('protien is None')
                    pdb_list2.append(pdb)
                    continue
                
                protein_list.append(protein)
                pdb_name.append(pdb)
                # print('inside protein')
                # print(protein)
            except FileNotFoundError as e:
                # Handle the FileNotFoundError here
                pdb_list3.append(pdb)
                print(f"File not found: {e}")

        print(f'pdb2 {pdb_list2}')
        print(f'pdb3 {pdb_list3}')
        return protein_list, pdb_list2, pdb_list3, pdb_name
    


    # def get_label(self):
    #     label_list = []
        
    #     for pdb in self.pdbs:
    #         pdb_path = os.path.join(self.pdb_folder, pdb)
            
    #         try:
    #             with open(pdb_path, 'r') as f:
    #                 line = f.readlines()[0]
    #                 label = self.label_dict[line[12:].strip()]
    #                 if pdb not in self.pdb_list2:
    #                     print('Inside label')
    #                     print(pdb)
    #                     label_list.append(label)
    #         except FileNotFoundError:
    #             print(f"File not found for PDB: {pdb}.pdb. Skipping...")
        
    #     return label_list
    def get_label(self):
        label_list = []

        for pdb in self.pdbs:
            if pdb not in self.pdb_list2 and pdb not in self.pdb_list3:
                # label = self.trans_df.loc[ self.trans_df['pdbid'] + '.pdb' == pdb, 'membrane_name_cache'].values
                label = self.trans_df.loc[ self.trans_df['pdbid']  == pdb, 'membrane_name_cache'].values
                # print('inside label')
                # print(label)
                # print(pdb)
                label = self.label_dict[label[0]] # (['abc']) that's why index [0]
                # print('inside 2 label')
                # print(label)
                label_list.append(label)


        return label_list
    
    def split(self):
        offset = 0
        splits = []

        train_size = int(self.num_samples * 0.75)
        val_size = int(self.num_samples * 0.24)

        train_split = torch_data.Subset(self, range(offset, offset + train_size))
        splits.append(train_split)

        offset += train_size
        val_split = torch_data.Subset(self, range(offset, offset + val_size))
        splits.append(val_split)

        offset += val_size
        test_split = torch_data.Subset(self, range(offset, self.num_samples))
        splits.append(test_split)

        return splits
    def stratified_k_fold_split(self, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        val_idx = []
        org_val_idx = []
        folds = []
        labels = self.targets["label"]
        # labels = [item[0] for item in labels] 

        original_indices = list(self.trans_df["original_index"])



        for train_indices, val_indices in skf.split(range(len(self.data)), labels):
           

              # Store original indices
            
            # print('self.trans_df["original_index"]')   
            # print(original_val_indices)

            # original_val_idx = self.trans_df.loc[val_indices, 'original_index']
            # print(original_val_idx)
            train_subset = torch_data.Subset(self, train_indices)
            val_subset = torch_data.Subset(self, val_indices)
            # name_subet = 

            folds.append((train_subset, val_subset))
            original_val_indices = [original_indices[i] for i in  val_indices]
            val_idx.append(val_indices)
            org_val_idx.append(original_val_indices)
            print('original_val')
            print(original_val_indices)

        return folds, val_idx, org_val_idx

    # def stratified_k_fold_split(self, n_splits=5):
    #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    #     folds = []
    #     labels = self.targets["label"]

    #     for train_indices, val_indices in skf.split(range(len(self.data)), labels):
    #         train_subset = torch_data.Subset(self, train_indices)
    #         val_subset = torch_data.Subset(self, val_indices)

    #         folds.append((train_subset, val_subset))

    #     return folds

    def __getitem__(self,index):
   
        protein=self.data[index]
        label=self.label_list[index]
        name = self.pdb_name[index]
        item={'graph':protein,'label':label, 'name':name}

        # print(pdb_path)
        if self.transform:
            item = self.transform(item)
        
        return item
       
    
    def __len__(self):
        return len(self.data)
    
    




    
csv_path = '/ocean/projects/bio230029p/bpokhrel/data/proteins-2023-04-15.csv'





# train_set, valid_set, test_set = dataset.split()

## if split function is not used
# lengths = [int(0.5 * len(dataset)), int(0.4 * len(dataset))]
# lengths += [len(dataset) - sum(lengths)]
# train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

'''Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 2)` or :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out) or (node_in, node_out, relation).
        edge_weight (array_like, optional): edge weights of shape :math:`(|E|,)`
        num_node (int, optional): number of nodes.
            By default, it will be inferred from the largest id in `edge_list`
        num_relation (int, optional): number of relations
        node_feature (array_like, optional): node features of shape :math:`(|V|, ...)`
        edge_feature (array_like, optional): edge features of shape :math:`(|E|, ...)`
        graph_feature (array_like, optional): graph feature of any shape'''




# Function to initialize your model
def initialize_model():
    gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], 
                                  num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                  batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                        edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                        edge_feature="gearnet")
    
    task = tasks.PropertyPrediction(gearnet_edge, graph_construction_model=graph_construction_model, num_mlp_layer=3, num_class=9,
                                              task=['label'], criterion="ce", metric=["acc"] )
    
    return task




dataset=ProteinLoader(csv_path=csv_path, transform=transform, random_seed =42)



print('stratified')
folds, val_idx, org_val_idx = dataset.stratified_k_fold_split(n_splits=3)
# print((np.array(org_val_idx, dtype=object)).shape)
# print('original_val_idx')
# print(org_val_idx)
# folds = dataset.stratified_k_fold_split(n_splits=3)
pdb_name = dataset.pdb_name
# print('after dataset')
# print(len(dataset.pdb_name))
# print(pdb_name)
# print(len(org_val_idx[0]))


for fold_idx, (train_set, valid_set) in enumerate(folds):
    l1=[]
    l2 =[]
    print(f"Processing fold {fold_idx + 1}")


    # Initialize task and optimizer for this fold
    task = initialize_model()
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)

    # Create a solver engine
    solver = core.Engine(task, train_set, valid_set, None, optimizer,
                         gpus=[0, 1], batch_size=2)

    # _checkpoint = torch.load("model_weights/mc_gearnet_edge.pth")
    _checkpoint = torch.load("/ocean/projects/bio230029p/bpokhrel/MultiviewContrast_try_OPM50_m100.pth")["model"]
    checkpoint = {}
    for k, v in _checkpoint.items():
        if k.startswith("model.model"):
            checkpoint[k[6:]] = v
        else:
            checkpoint[k] = v
    checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("mlp")}
    task.load_state_dict(checkpoint, strict=False)

    # Training
    solver.train(num_epoch=3)
    solver.save('trim_e250_15_a.pth')

    # Validation
    solver.evaluate("valid")
    # fold_pdbnames = {}
    # fold_labels = []
    # for i in valid_set:
    #     fold_pdbnames[i['name']] = i['label']
        # fold_labels.append(i['label'])
    
    # for val in valid_set:
    # # print(val)
    #     l1.append(val['name'])

    # for i in org_val_idx[fold_idx]:
    #     l2.append(dataset.trans_df.loc[dataset.trans_df['original_index'] == i, 'pdbid'].values[0])
    # # print('l1')
    # print(l1[:10])
    # print('l2')
    # print(l2[:10])
    #     for j in i:
    #         # print(i,j)
    #         # print(pdb_name[j])
    #         print('inside org_val_idx')
    #         print(dataset.trans_df['pdbid'][dataset.trans_df['original_index'] == j])
    #         fold_pdbnames.append(dataset.trans_df['pdbid'][dataset.trans_df['original_index'] == j])

    

    # base_path = "/ocean/projects/bio230029p/bpokhrel/names"

    # file_prefix = "oo1p"
    # file_number = 1
    # while True:
    #     file_name = f"{file_prefix}{file_number}.npy"
    #     file_path = os.path.join(base_path, file_name)

    #     if not os.path.exists(file_path):
    #         # Files with these names do not exist, so save the data with these names
    #         np.save(file_path, l2)
    #         # np.save(target_file_path, all_target_values)
    #         break  # Exit the loop

        
    #     file_number += 1