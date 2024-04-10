from sklearn.metrics import mean_absolute_error,mean_squared_error
from scipy.stats import pearsonr
import copy
import torch
from torch import Tensor

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return mae, pearson_corr

class Vail_When_Train(object):

    def __init__(self, model, superdict):
        torch.cuda.empty_cache()
        sec_model = copy.deepcopy(model).eval()
        sec_model = sec_model.to(superdict["device"])
        self.model = sec_model
        self.device = superdict["device"]

    def vaild(self,valid_dataloader):
        label_list = []
        pre_list = []
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                proten_graph, chem_graph, label, sequence = data
                proten_graph = proten_graph.to(self.device)
                chem_graph = chem_graph.to(self.device)
                label = float(label.item())

                sequence = sequence.to(self.device)
                data = [proten_graph, chem_graph, sequence]
                try:
                    pre = self.model.forward(data)
                    pre = float(pre.item())
                    label_list.append(label)
                    pre_list.append(pre)
                except Exception as e:
                    print(e)
                    continue

            mae, pearson_corr = calculate_metrics(label_list, pre_list)
        return mae, pearson_corr


    def vaild_All(self,valid_dataloader):
        label_list = []
        pre_list = []
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                proten_graph, chem_graph, label, sequence = data
                proten_graph = proten_graph.to(self.device)
                chem_graph = chem_graph.to(self.device)
                label = float(label.item())

                sequence = sequence.to(self.device)
                data = [proten_graph, chem_graph, sequence]
                try:
                    pre = self.model.forward(data)
                    pre = float(pre.item())
                    label_list.append(label)
                    pre_list.append(pre)
                except Exception as e:
                    print(e)
                    continue

            mae, pearson_corr = calculate_metrics(label_list, pre_list)
        rmse = mean_squared_error(label_list, pre_list, squared=False)
        return rmse,mae, pearson_corr

if __name__ =='__main__':
    import pickle
    import os
    folders = "5078/01_23_14_43/modules/"
    pickle_path = os.path.join(folders,"01_23_14_43___database_pdbbindlr_6.5e-05__batch_size_8__output_channel_1024__pad_size_(1200, 300)__mean_0__kernel_size_(4, 3)__p_0.1__L1__Adam_all.pth")
    super_dict_path = os.path.join(folders,"super_dict.pickle")

    with open(super_dict_path,"rb") as file:
        super_dict = pickle.load(file)
    cuda = super_dict["device"]
    print(cuda)

    module = torch.load(pickle_path,map_location=cuda)

    from pyg_dataloader_transformer_pdbbind import *
    from torch_geometric.loader import DataLoader

    test_vaild_path = "CASF.pickle"
    test_pickles = Read_Pick("", test_vaild_path)
    test_pickles.deal_all_graph(16,74)
    test_df = test_pickles.all_data
    test_dataset = Df_To_Dataset(test_df, 0)
    test_dataloader = DataLoader(test_dataset)

    try:
        vailder = Vail_When_Train(module, super_dict)
        rmse,mae, pearson_corr = vailder.vaild_All(test_dataloader)
        print(f"rmse_{rmse}")
        print(f"mae_{mae}")
        print(f"pr_{pearson_corr}")
    except Exception as e:
        print(e)
