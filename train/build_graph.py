import numpy as np
import pandas as pd
import re

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, PandasTools

from func_timeout import func_set_timeout


def deal_node(Node_df, feature_df):
    Tmp_list = []
    Position_X = Node_df["x"]
    Position_Y = Node_df["y"]
    Position_Z = Node_df["z"]
    Node_df["x"] = Node_df["x"]-Position_X.mean()
    Node_df["y"] = Node_df["y"]-Position_Y.mean()
    Node_df["z"] = Node_df["z"]-Position_Z.mean()
    for i in Node_df.index:
        Id = Node_df.loc[i]["Id"]
        aa = Node_df.loc[i]["Name"]
        feeture = feature_df[feature_df['name'] == aa]['feature_onehot'].to_list()[
            0]
        feeture = str(feeture)
        feature_onehot = re.findall(r'[0-1]', feeture)
        posion = [Node_df.loc[i]['x'], Node_df.loc[i]
                  ['y'], Node_df.loc[i]['z']]
        feature = feature_onehot+posion
        if len(feature) == 16:

            feature = [0, 0, 0]+feature
        for j in range(len(feature)):
            feature[j] = float(feature[j])
        Tmp_list.append(pd.DataFrame(
            [[Id, feature]], columns=['id', 'feature']))
    Export_df = pd.concat(Tmp_list, ignore_index=True)
    return(Export_df)


def BuildDistanceNp(Node_df):
    results_list = []
    CON = [0]
    Helix = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Sheet = [0, 0, 0]
    Els = [0]
    Distance = [0]
    High = CON+Helix+Sheet+Els
    for loc_i in range(len(Node_df.index)):
        i = Node_df.index[loc_i]
        first_id = int(Node_df.loc[i]['id'])
        first_vector = np.array(Node_df.loc[i]['feature'][-3:])
        for loc_j in range(loc_i, len(Node_df.index)):
            j = Node_df.index[loc_j]
            second_id = int(Node_df.loc[j]['id'])
            if abs(second_id-first_id) < 3:
                continue
            else:
                second_vector = np.array(Node_df.loc[j]['feature'][-3:])
                distance = np.linalg.norm(first_vector-second_vector)
                if distance > 10:
                    continue
                else:
                    Distance = [1]
                    feature = CON+Helix+Sheet+Els+Distance
                    results_list.append(pd.DataFrame(
                        [[first_id, second_id, feature]], columns=['src', 'dst', 'feature']))
    Result_df = pd.concat(results_list, ignore_index=True)
    return(Result_df)


def deal_edge(Edge_df, Node_df):
    result = []
    tmp_results = []
    for i in Edge_df.index:
        src = Edge_df.loc[i]["src"]
        dst = Edge_df.loc[i]["dst"]
        CON = [0]
        Helix = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Sheet = [0, 0, 0]
        Els = [0]
        Distance = [0]
        # CON Helix start end Sheet start end else
        Edge_type_Meta = Edge_df.loc[i]["type"]
        if Edge_type_Meta == 'CON':
            CON[0] = 1
        elif Edge_type_Meta.split('_')[0] == 'HELIX':
            Helix[0] = 1
            Helix_type = int(Edge_type_Meta.split('_')[2])+2
            Helix[Helix_type] = 1

            Helix_number = int(Edge_type_Meta.split('_')[1])
            try:
                last_Edge_type_Meta = Edge_df.loc[i-1]["type"]
                last_num = int(last_Edge_type_Meta.split('_')[1])
            except:
                last_num = 0
            try:
                Next_Edge_type_Meta = Edge_df.loc[i-1]["type"]
                next_num = int(Next_Edge_type_Meta.split('_')[1])
            except:
                next_num = 0
            if Helix_number != last_num:
                Helix[1] = 1
                pass
            if Helix_number != next_num:
                Helix[2] = 1
                pass

        elif Edge_type_Meta.split('_')[0] == 'SHEET':
            Sheet[0] = 1
            sheet_type = int(Edge_type_Meta.split('_')[3])
            Sheet[sheet_type] = 1
        else:
            Els[0] = 1
        all_feature = CON+Helix+Sheet+Els+Distance
        tmp_results.append(pd.DataFrame(
            [[[src, dst], all_feature, str(src)+"+"+str(dst)]], columns=['src-dst', 'feature', 'label']))

    tmp_df = pd.concat(tmp_results, ignore_index=True)

    label_list = set(tmp_df['label'].to_list())
    for label in label_list:
        t_df = tmp_df[tmp_df['label'] == label].reset_index(drop=True)
        edge = t_df['src-dst'][0]
        if len(t_df) == 1:
            result.append(pd.DataFrame(
                [[edge[0], edge[1], t_df['feature'][0]]], columns=['src', 'dst', 'feature']))
        elif len(t_df) > 1:
            feature_s_list = t_df['feature'].to_numpy().tolist()
            result_feature = np.array(feature_s_list).sum(axis=0).tolist()
            
            result.append(pd.DataFrame(
                [[edge[0], edge[1], result_feature]], columns=['src', 'dst', 'feature']))

    result.append(BuildDistanceNp(Node_df))
    Result_pd = pd.concat(result, ignore_index=True)
    return(Result_pd)


def Deal_All(Node_df, Edge_df, Feature_df):

    Node_df_Result = deal_node(Node_df, Feature_df)
    Edge_df_Result = deal_edge(Edge_df, Node_df_Result)

    ID_Sort_List = list(set(Node_df_Result["id"]))
    ID_Sort_List.sort()
    ID_Sort_Dict = {i: ID_Sort_List.index(i) for i in ID_Sort_List}

    All_Node_Results_List = []
    All_Edge_Results_List = []

    for i in Node_df_Result.index:
        Id = Node_df_Result.loc[i]["id"]
        Feature = Node_df_Result.loc[i]["feature"]
        All_Node_Results_List.append(pd.DataFrame(
            [[ID_Sort_Dict[Id], Feature]], columns=["id", "feature"]))

    for i in Edge_df_Result.index:
        #'src', 'dst', 'feature'
        src = Edge_df_Result.loc[i]["src"]
        dst = Edge_df_Result.loc[i]["dst"]
        feature = Edge_df_Result.loc[i]["feature"]
        try:
            All_Edge_Results_List.append(pd.DataFrame(
                [[ID_Sort_Dict[src], ID_Sort_Dict[dst], feature]], columns=['src', 'dst', 'feature']))
        except Exception as e:
            continue

    All_Edge_Results_df = pd.concat(All_Edge_Results_List, ignore_index=True)
    All_Node_Results_df = pd.concat(All_Node_Results_List, ignore_index=True)
    All_Node_Results_df = All_Node_Results_df.sort_values(by=['id'])

    return All_Node_Results_df, All_Edge_Results_df,ID_Sort_Dict


# Chem Graph Build

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk_int(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: int(x == s), allowable_set))


def GetAtomFeatures(atom):
    results = one_of_k_encoding_unk(atom.GetSymbol(),
                                    ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    results = results + \
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    results = results + \
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    results = results + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    results = results + one_of_k_encoding_unk(atom.GetHybridization(),
                                              [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])
    results = results + [atom.GetIsAromatic()]
    results = results + \
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    return(results)


def Get3DFeatureFromMol(mol1):
    m1_con = mol1.GetConformer(id=0)
    features = []
    for atom_index in range(mol1.GetNumAtoms()):
        atom = mol1.GetAtomWithIdx(atom_index)
        atom_feature = GetAtomFeatures(atom)
        postion = list(m1_con.GetAtomPosition(atom_index))
        atom_feature = atom_feature+postion  # 3D info
        features.append(atom_feature)
    np_feature = np.array(features, dtype=float)
    return(np_feature)

@func_set_timeout(15)
def Deal_InChI(InChI):
    mol = Chem.MolFromInchi(InChI)
    mol = Chem.AddHs(mol)
    ignore_flag1 = 0
    ignore1 = False
    while AllChem.EmbedMolecule(mol) == -1:
        print('retry')
        ignore_flag1 = ignore_flag1 + 1
        if ignore_flag1 >= 5:
            ignore1 = True
            break
    if ignore1:
        raise TypeError
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    feature_numpy = Get3DFeatureFromMol(mol)
    adjacency_numpy = Chem.GetAdjacencyMatrix(mol)
    return adjacency_numpy, feature_numpy

@func_set_timeout(15)
def Deal_Mol(mol):
    mol = Chem.AddHs(mol)
    ignore_flag1 = 0
    ignore1 = False
    while AllChem.EmbedMolecule(mol) == -1:
        print('retry')
        ignore_flag1 = ignore_flag1 + 1
        if ignore_flag1 >= 5:
            ignore1 = True
            break
    if ignore1:
        raise TypeError
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    feature_numpy = Get3DFeatureFromMol(mol)
    adjacency_numpy = Chem.GetAdjacencyMatrix(mol)
    return adjacency_numpy, feature_numpy

# sequences to matrix

def Sequences_To_Matrix(node_df):
    AA_List = [
    "alanine",
    "arginine",
    "asparagine",
    "aspartic acid",
    "cysteine",
    "glutamine",
    "glutamic acid",
    "glycine",
    "histidine",
    "isoleucine",
    "leucine",
    "lysine",
    "methionine",
    "phenylalanine",
    "proline",
    "serine",
    "threonine",
    "tryptophan",
    "tyrosine",
    "valine",
    "unknown"
    ]
    sequences = node_df["Name"].tolist()
    mtrix_list = [one_of_k_encoding_unk_int(se,AA_List) for se in sequences]
    matrix = np.array(mtrix_list)
    return matrix