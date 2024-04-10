import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from scipy.special import log_softmax,softmax
import numpy as np
from torch_geometric.data import Data

class Read_Pick():
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


    def label_onhot(self,label):
        max = int(np.log10(np.max(label)))
        ou_np = np.zeros((len(label),max+2))
        for i in range(len(label)):
            tmp = np.zeros(max+2)
            if label[i] < 1:
                tmp[0] = label[i]
                ou_np[i] = tmp
            else:
                lo = int(np.log10(label[i]))
                tmp[lo+1] = label[i]/10**lo
                ou_np[i] = tmp
        return ou_np

    def softmax_label(self):
        label = self.all_data["label"].to_numpy()
        softmaxed = softmax(label)
        self.all_data["label"] = softmaxed
    
    def deal_all_graph(self,protein_channel,chem_channel):
        NullDatafarme = pd.DataFrame(columns=["Protein","Chem","label","sequence"])
        for i in range(self.count):
            protin_tmp = self.all_data.loc[i,"Protein"]
            protin_tmp_x = protin_tmp.x
            protin_tmp_x = protin_tmp_x[:,:protein_channel]
            new_protein_graph = Data(x=protin_tmp_x,edge_index=protin_tmp.edge_index,edge_attr=protin_tmp.edge_attr)
            NullDatafarme.loc[i,"Protein"] = new_protein_graph

            chem_tmp = self.all_data.loc[i,"Chem"]
            chem_tmp_x = chem_tmp.x
            chem_tmp_x = chem_tmp_x[:,:chem_channel]
            new_chem_graph = Data(x=chem_tmp_x,edge_index=chem_tmp.edge_index,edge_attr=chem_tmp.edge_attr)
            NullDatafarme.loc[i,"Chem"] = new_chem_graph

            NullDatafarme.loc[i,"label"] = self.all_data.loc[i,"label"]
            NullDatafarme.loc[i,"sequence"] = self.all_data.loc[i,"sequence"]
        
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


class Df_To_Dataset(Dataset):
    def __init__(self, data_frame,max):
        super().__init__()
        print(data_frame.shape)
        self.length = len(data_frame["label"])-1
        self.label = torch.Tensor(data_frame["label"])
        self.proten_graph = data_frame["Protein"]
        self.chem_graph = data_frame["Chem"]
        self.sequence = data_frame["sequence"]

    def return_mean(self):
        label_mean = self.label.mean()
        return(label_mean)

    def __len__(self):
        return self.length

    def __getitem__(self, index) :
        try:
            return [self.proten_graph[index], self.chem_graph[index], self.label[index],torch.Tensor(self.sequence[index])]
        except IndexError:
            pass
    
