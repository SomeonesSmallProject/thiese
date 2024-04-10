from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd
import pubchempy as pcp

Feature_dict = {
    0: ("NC(C)C(=O)O", 0),
    1: ("[#6]-[#16]-[#6]", 0),  # C-S-C
    2: ("[#6]-[#16]", 0),  # C-S
    3: ("[D2][#6](=[#8])~[!#6]", 0),  # -C(=O)- debug
    4: ("[!#6]CCCCC[!#6]", 0),  # len(C)
    5: ("CC(=O)N", 0),
    6: ("c1ccccc1", 0),  # PAr
    7: ("[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]1[!#6]", 0),  # Ar+X
    8: ("[#7]1~*~*~*~*~1", 0),  # N5
    9: ("[#7][#6](~[#6])[#6](~[#8])[#8]", 0),  # Pro
    10: ("[#7][#6](=[#7])[#7]", 0), 
    11: ("CC(O)C", 0),
    12: ("[D3][D3][D3]", 0),
    13: ("[D3][D3][D2]", 0),
    14: ("O~CCC~O", 0),
    15: ("[D3][D3][D2]O", 0)
}


def ListToString(tolist):
    st = ""
    for i in range(len(tolist)):
        st = st+str(tolist[i])
    return(st)


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
    "valine"
]

MetaKey = []
for i in range(len(Feature_dict.keys())):
    MetaKey.append("0")

df_list = []

for aa in AA_List:
    print(aa)
    key = MetaKey.copy()
    compound = Chem.MolFromSmiles(
        pcp.get_compounds(aa, "name")[0].isomeric_smiles)
    for i in Feature_dict.keys():
        (patt, count) = Feature_dict[i]
        key[i] = str(int(compound.HasSubstructMatch(Chem.MolFromSmarts(patt))))
    feature = ListToString(key)
    df_list.append(
        pd.DataFrame(
            [[aa, str(int(feature, 2)),feature]],
            columns=["name", "feature","feature_onehot"]
        )
    )

df_all = pd.concat(df_list)
df_all.to_csv("feature.csv",index=False)
