import torch
from pyg_Module_Chem import *
from pyg_dataloader_transformer_tdc import *
from pyg_explan_Deal_Data_dm import *

def smi_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem_Adj, Chem_Feature = Deal_Mol(mol)
    Chem_Adj = torch.Tensor(Chem_Adj)
    Chem_Feature = torch.Tensor(Chem_Feature)
    edge_index = Chem_Adj.nonzero().t().contiguous()
    Chem_graph = Data(x=Chem_Feature, edge_index=edge_index)
    return Chem_graph

def remove_node_data(data:Data,node_to_remove:int):
    node_num = data.num_nodes
    data.edge_index = data.edge_index.to(torch.long)
    node_Tensor = torch.tensor([n for n in range(node_num) if n != node_to_remove], dtype=torch.long)
    sub_data = data.subgraph(node_Tensor)
    return sub_data

def explain_test(module:Modules,data:list):
    with torch.no_grad():
        mol:Data = data[0]
        out = module.forward([mol])
        chem_node_num = mol.num_nodes
        Chem_dict = {}

        for i in range(chem_node_num):
            mol_copy = mol.clone()
            tmp_mol = remove_node_data(mol_copy,i)
            try:
                tmp_out = module.forward([tmp_mol])
                diff = out - tmp_out
                print(f"{i}_{diff}")
                Chem_dict[i] = diff
            except:
                continue
        return Chem_dict


if __name__ =='__main__':
    module_path = "03_17_16_33___database_pdbbindlr_7e-06__batch_size_8__output_channel_4096__pad_size_(300,)__mean_0__kernel_size_(4, 3)__p_0.1__mse__Adam_44.pth"
    module = torch.load(module_path, map_location=torch.device('cpu'))
    module.eval()

    smiles="CC(=O)OC1=CC=CC=C1C(=O)O"
    data = [smi_to_graph(smiles)]
    Chem_dict = explain_test(module,data)