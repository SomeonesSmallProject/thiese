import pandas as pd
import sqlite3
import os
import re
from build_graph import *
import torch
from torch_geometric.data import Data
import numpy as np
import multiprocessing
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import time
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import subprocess
#import pyjion
#pyjion.config(pgc=False,level=2,threshold=4)
#pyjion.enable()

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def Deal_Mol_Error(File_Name, Commit_SQL, e, Protein_Pass_Name, Protein):
    with open(File_Name, "a") as Fn:
        Fn.writelines(Commit_SQL + "\n")
        Fn.writelines(str(e) + "\n")

    with open(Protein_Pass_Name, "a") as Ppn:
        Ppn.writelines(Protein + "\n")


def list_to_string(input_list):
    r = ""
    for i in range(len(input_list)):
        r = r + str(input_list[i]) + "-"
    r = r[:-1]
    return r


def string_to_list(input_string):
    return input_string.split("-")


def random_matrix(Pro_num, Com_num):
    Max_len = max((Pro_num * Com_num), (2 ** (len(bin(Com_num)))))
    Min_lin = min((2 ** (len(bin(Pro_num * Com_num)) - 2)), (2 ** (len(bin(Com_num)))))
    flag = 1
    while flag < 10:
        Random_matrix_Max_Meta = np.random.randint(
            [Pro_num, Com_num], size=[Max_len, 2]
        ).tolist()
        Random_matrix_Max_List = [list_to_string(i) for i in Random_matrix_Max_Meta]
        Random_matrix_Max_ListSet = list(set(Random_matrix_Max_List))
        if len(Random_matrix_Max_ListSet) >= Min_lin:
            Random_matrix_Max_ListSet = Random_matrix_Max_ListSet[:Min_lin]
            break
        flag = flag + 1
    if len(Random_matrix_Max_ListSet) < Min_lin:
        Random_matrix_Max_ListSet = Random_matrix_Max_ListSet[
            : 2 ** (len(bin(len(Random_matrix_Max_ListSet) - 2)))
        ]
    Random_matrix_List = [string_to_list(i) for i in Random_matrix_Max_ListSet]
    return Random_matrix_List


def Folder_Graph_Build(folder):
    Feature_df = pd.read_csv("feature.csv")
    node_file_path = os.path.join(folder, "node.csv")
    edge_file_path = os.path.join(folder, "edge.csv")
    node_df = pd.read_csv(node_file_path)
    edge_df = pd.read_csv(edge_file_path)
    if len(node_df) < 50 or len(edge_df) < 50:
        raise ValueError
    node_df_result, edge_df_result,ID_Sort_Dict = Deal_All(node_df, edge_df, Feature_df)
    sequence_matrix = Sequences_To_Matrix(node_df)

    src_list = edge_df_result["src"].to_list()
    src_numpy = np.array(src_list, dtype=np.float64)
    dst_list = edge_df_result["dst"].to_list()
    dst_numpy = np.array(dst_list, dtype=np.float64)
    edge_index_numpy = np.array([src_numpy, dst_numpy])
    Node_feature_list = list(node_df_result["feature"].to_list())
    Node_feature_numpy = np.array(Node_feature_list, dtype=np.float64)
    Node_feature = torch.Tensor(Node_feature_numpy)
    try:
        Edge_feature_list = list(edge_df_result["feature"].to_list())
        Edge_feature_numpy = np.array(Edge_feature_list, dtype=np.float64)
        Edge_feature = torch.Tensor(Edge_feature_numpy)

        data = Data(
            x=Node_feature,
            edge_index=torch.Tensor(edge_index_numpy),
            edge_attr=Edge_feature,
        )
    except Exception as e:
        data = Data(x=Node_feature, edge_index=torch.Tensor(edge_index_numpy))

    del Feature_df
    del node_df
    del edge_df
    print(folder)
    return [data, sequence_matrix]


def Folder_Graph_Build_Mult(folder):

    Protein_graph = Folder_Graph_Build(folder)
    tmp_df = pd.DataFrame(
            [Protein_graph], columns=["Protein_graph", "sequence_matrix"]
        )

    return tmp_df


@func_set_timeout(20)
def InchiToGraph(InchI, label):
    try:
        Chem_Adj, Chem_Feature = Deal_InChI(InchI)
        Chem_Adj = torch.Tensor(Chem_Adj)
        Chem_Feature = torch.Tensor(Chem_Feature)
        edge_index = Chem_Adj.nonzero().t().contiguous()
        Chem_Df = pd.DataFrame(
            [[Data(x=Chem_Feature, edge_index=edge_index), InchI, label]],
            columns=["Chem", "InchI", "label"],
        )
    except:
        Chem_Df = None
    return Chem_Df


@func_set_timeout(20)
def MolFileToGraph(Mol_file_path, label):
    try:
        mol = Chem.MolFromMol2File(Mol_file_path)
        Chem_Adj, Chem_Feature = Deal_Mol(mol)
        Chem_Adj = torch.Tensor(Chem_Adj)
        Chem_Feature = torch.Tensor(Chem_Feature)
        edge_index = Chem_Adj.nonzero().t().contiguous()
        Chem_Df = pd.DataFrame(
            [[Data(x=Chem_Feature, edge_index=edge_index), label]],
            columns=["Chem", "label"],
        )
    except Exception as e:
        Chem_Df = None
    return Chem_Df


@func_set_timeout(20)
def SmilesToGraph(smiles, label):
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem_Adj, Chem_Feature = Deal_Mol(mol)
        Chem_Adj = torch.Tensor(Chem_Adj)
        Chem_Feature = torch.Tensor(Chem_Feature)
        edge_index = Chem_Adj.nonzero().t().contiguous()
        Chem_Df = pd.DataFrame(
            [[Data(x=Chem_Feature, edge_index=edge_index), label]],
            columns=["Chem", "label"],
        )
    except Exception as e:
        Chem_Df = None
    return Chem_Df


def download_pdb(pdb_ID, output_dir):
    subprocess.run(["Download_PDB", pdb_ID, output_dir])



if __name__ == "__main__":
    print("Start")
    pwd = os.getcwd()
    data_csv_file_path = "CASF-2016/power_screening/CoreSet.dat"
    csv_df = pd.read_csv(data_csv_file_path, sep="   ")
    df = csv_df[["#code", " logKa"]]
    df = df.rename(columns={" logKa": "logKa"})
    mol_files_meta_folder = "CASF-2016/coreset"
    output_folder = "output"
    Export_df_list = []

    for i in df.index:
        pdb_id = df.loc[i]["#code"].strip()
        logKa = df.loc[i]["logKa"]
        mol_file_path = os.path.join(mol_files_meta_folder, pdb_id,pdb_id + "_ligand.mol2")
        pdb_df = Folder_Graph_Build_Mult(os.path.join(output_folder, pdb_id))

        ligand_df = MolFileToGraph(mol_file_path, logKa)

        protein_graph = pdb_df["Protein_graph"][0]
        sequence_matrix = pdb_df["sequence_matrix"][0]
        ligand_graph = ligand_df["Chem"][0]
        label = ligand_df["label"][0]

        Export_df_list.append(
            pd.DataFrame(
                [[protein_graph, ligand_graph, label, sequence_matrix]],
                columns=["Protein", "Chem", "label", "sequence"],
            )
        )

    Export_df = pd.concat(Export_df_list, ignore_index=True)
    Export_name = os.path.join(output_folder, "/Export.pickle")
    Export_df.to_pickle(Export_name)
