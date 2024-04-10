import torch
from torch_geometric.data import Dataset
import pandas as pd
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

class Read_tdc_Pick():
    def __init__(self, folder, file_name):
        super().__init__()
        AllDatabase_meta = pd.read_pickle(os.path.join(folder, file_name))
        AllDatabase=AllDatabase_meta
        max_label = max(AllDatabase['label'])
        min_label = min(AllDatabase['label'])
        print(f"max_{max_label}__min_{min_label}")
        AllDatabase = AllDatabase.reset_index(drop=True)
        self.count = len(AllDatabase["label"])
        self.all_data = AllDatabase.reset_index(drop=True)
        if int(max_label/min_label)>10:
            self.log_label()
            print("log_label")

    def log_label(self):
        self.all_data["label"] = np.log(self.all_data["label"])


    def deal_all_graph(self,chem_channel):
        NullDatafarme = pd.DataFrame(columns=["Chem","label"])
        for i in range(self.count):
            chem_tmp = self.all_data.loc[i,"Chem"]
            chem_tmp_x = chem_tmp.x
            chem_tmp_x = chem_tmp_x[:,:chem_channel]
            new_chem_graph = Data(x=chem_tmp_x,edge_index=chem_tmp.edge_index,edge_attr=chem_tmp.edge_attr)
            NullDatafarme.loc[i,"Chem"] = new_chem_graph

            NullDatafarme.loc[i,"label"] = self.all_data.loc[i,"label"]
        
        self.all_data = NullDatafarme

    def return_vt_part(self):
        AllDatabase = self.all_data
        test_dataset = AllDatabase.sample(frac=0.8)
        valid_dataset = AllDatabase[~AllDatabase.index.isin(test_dataset.index)]
        valid_dataset = valid_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)
        return valid_dataset, test_dataset

    def return_all(self):
        AllDatabase = self.all_data
        train_dataset = AllDatabase.sample(frac=0.8)
        else_dataset = AllDatabase[~AllDatabase.index.isin(
            train_dataset.index)]
        valid_dataset = else_dataset.sample(frac=0.5)
        test_dataset = else_dataset[~else_dataset.index.isin(valid_dataset)]
        self.train_dataset = train_dataset.reset_index(drop=True)
        self.valid_dataset = valid_dataset.reset_index(drop=True)
        self.test_dataset = test_dataset.reset_index(drop=True)
        return self.train_dataset, self.valid_dataset, self.test_dataset


class tdc_Df_To_Dataset(Dataset):
    def __init__(self, data_frame,max):
        super().__init__()
        data_frame = data_frame[data_frame[['Chem', 'label']].notnull().all(axis=1)]
        data_frame = data_frame.reset_index(drop=True)
        print(data_frame.shape)
        self.length = len(data_frame["label"])-1
        self.label = torch.Tensor(data_frame["label"])
        self.chem_graph = data_frame["Chem"]

    def return_mean(self):
        label_mean = self.label.mean()
        return(label_mean)

    def __len__(self):
        return self.length
    
    def len(self):
        return self.length
    
    def get(self, idx: int):
        try:
            chem_graphs = self.chem_graph[idx]
            labels = self.label[idx]
            return chem_graphs, torch.Tensor(labels)
        except IndexError:
            pass

#    def __getitem__(self, idx) :
#        try:
#            chem_graphs = [self.chem_graph[i] for i in idx]
#            labels = [self.label[i] for i in idx]
#            mols = Data.batch(chem_graphs)
#            return mols, torch.Tensor(labels)
#        except IndexError:
#            pass
    
