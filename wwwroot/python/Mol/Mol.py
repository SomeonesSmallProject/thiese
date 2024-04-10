import torch
from pyg_Module_Chem import *
from pyg_dataloader_transformer_tdc import *
from pyg_explan_Deal_Data_dm import *
import numpy as np
import rdkit
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image, ImageOps

from func_timeout import FunctionTimedOut
import argparse
import json

def _drawerToImage(d2d):
    try:
        import Image
    except ImportError:
        from PIL import Image
    sio = BytesIO(d2d.GetDrawingText())
    return Image.open(sio)

def clourMol(mol,highlightAtoms_p=None,highlightAtomColors_p=None,highlightBonds_p=None,highlightBondColors_p=None,sz=[1600,1600]):
    '''

    '''
    d2d = rdMolDraw2D.MolDraw2DCairo(sz[0], sz[1],sz[0], sz[1])
    op = d2d.drawOptions()
    op.FontScale = 1.5
    op.addAtomIndices = True
    #op.dotsPerAngstrom = 20
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p,highlightAtomColors=highlightAtomColors_p, highlightBonds= highlightBonds_p,highlightBondColors=highlightBondColors_p)
    d2d.FinishDrawing()
    product_img=_drawerToImage(d2d)
    return product_img
def StripAlphaFromImage(img):
    '''This function takes an RGBA PIL image and returns an RGB image'''

    if len(img.split()) == 3:
        return img
    return Image.merge('RGB', img.split()[:3])

def TrimImgByWhite(img, padding=10):
    '''This function takes a PIL image, img, and crops it to the minimum rectangle
    based on its whiteness/transparency. 5 pixel padding used automatically.'''

    # Convert to array
    as_array = np.array(img)  # N x N x (r,g,b,a)

    # Set previously-transparent pixels to white
    if as_array.shape[2] == 4:
        as_array[as_array[:, :, 3] == 0] = [255, 255, 255, 0]

    as_array = as_array[:, :, :3]

    # Content defined as non-white and non-transparent pixel
    has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
    xs, ys = np.nonzero(has_content)

    # Crop down
    margin = 5
    x_range = max([min(xs) - margin, 0]), min([max(xs) + margin, as_array.shape[0]])
    y_range = max([min(ys) - margin, 0]), min([max(ys) + margin, as_array.shape[1]])
    as_array_cropped = as_array[
        x_range[0]:x_range[1], y_range[0]:y_range[1], 0:3]

    img = Image.fromarray(as_array_cropped, mode='RGB')

    return ImageOps.expand(img, border=padding, fill=(255, 255, 255))

def smi_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem_Adj, Chem_Feature = Deal_Mol(mol)
    Chem_Adj = torch.Tensor(Chem_Adj)
    Chem_Feature = torch.Tensor(Chem_Feature)
    edge_index = Chem_Adj.nonzero().t().contiguous()
    Chem_graph = Data(x=Chem_Feature, edge_index=edge_index)
    return Chem_graph

def remove_node_data(data:Data,node_to_remove:int,device):
    node_num = data.num_nodes
    data.edge_index = data.edge_index.to(torch.long)
    node_Tensor = torch.tensor([n for n in range(node_num) if n != node_to_remove], dtype=torch.long)
    node_Tensor = node_Tensor.to(device)
    sub_data = data.subgraph(node_Tensor)
    return sub_data

def explain(module:Modules,data:list,device):
    with torch.no_grad():
        mol:Data = data[0]
        mol = mol.to(device)
        out = module.forward([mol])
        chem_node_num = mol.num_nodes
        Chem_dict = {}
        diff_list = []

        for i in range(chem_node_num):
            mol_copy = mol.clone()
            tmp_mol = remove_node_data(mol_copy,i,device)
            tmp_mol = tmp_mol.to(device)
            try:
                tmp_out = module.forward([tmp_mol])
                diff = out - tmp_out
                diff = diff.cpu()
                #print(f"{i}_{diff}")
                Chem_dict[i] = float(diff)
                diff_list.append(float(diff))
            except:
                continue
        diff_array = np.array(diff_list)
        diff_list_sof = 7*((diff_array-min(diff_array))/max(diff_array-min(diff_array)))
        tmp_dict = {}
        for i in range(len(diff_list_sof)):
            tmp_dict[diff_list[i]] = diff_list_sof[i]
        for i in Chem_dict.keys():
            Chem_dict[i] = int(tmp_dict[Chem_dict[i]])
        return out,Chem_dict

if __name__=="__main__":
    colormap = plt.get_cmap('coolwarm')
    tmp = np.linspace(0, 1, 8)
    colors   = colormap(tmp)
    color_dict = {}
    for i,color in enumerate(colors):
        color_dict[i] = (list(color[:-1]))

    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--pickle', dest='pickle_path', required=True, help='Path to pickle file')
    parser.add_argument('--smiles', dest='smiles', help='SMILES string')
    parser.add_argument('--file', dest='file_path', help='Path to file')
    args = parser.parse_args()

    module_path = args.pickle_path
    smiles = args.smiles
    file = args.file_path

    device = torch.device('cpu')
    module = torch.load(module_path, map_location=device)
    result = {}
    try:
        data = [smi_to_graph(smiles)]
        out,Chem_dict = explain(module,data,device)
        mol = rdkit.Chem.MolFromSmiles(smiles)
        atom_colors = {}
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom_idx in Chem_dict:
                atom_colors[atom_idx] = (color_dict[Chem_dict[atom_idx]][0],color_dict[Chem_dict[atom_idx]][1],color_dict[Chem_dict[atom_idx]][2]) 
        img = clourMol(mol,highlightAtoms_p=atom_colors.keys(),highlightAtomColors_p=atom_colors)
        #img = TrimImgByWhite(img)
        file_name = smiles.replace("/", "_").replace("\\", "_").replace(" ", "_")
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        result[f"{smiles}_image"] = img_bytes
        result[f"{smiles}_score"] = out
        json_string = json.dumps(result)
        print(json_string)
    except FunctionTimedOut as f:
        print(f"{f}_{smiles}")
    except Exception as e:
        print(f"{e}_{smiles}")
