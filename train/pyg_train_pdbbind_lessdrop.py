import torch
from pyg_Module_Transformer_lessdrop import *
from torch_geometric.loader import DataLoader
import pandas as pd
import os
from pyg_dataloader_transformer_pdbbind import *
import pickle
from pyg_vaild import *

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # torch.manual_seed(42)

    cuda = torch.device("cuda:2")

    folder = os.getcwd()
    print(folder)
    train_file_name = "../PDBbind/All_train.pickle"
    test_file_name = "../PDBbind/All_vaild.pickle"

    batch_size = 1

    wd = 0
    train_cucle = 256

    train_pickles = Read_Pick(folder, train_file_name)
    #train_pickles.softmax_label()
    """    train_df = train_pickles.all_data
    train_dataset = Df_To_Dataset(train_df, 0)
    train_dataloader = DataLoader(train_dataset)"""

    
    log_batch = 16
    tmp_time = time.localtime(time.time())
    runs_folder = "{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}".format(tmp_time.tm_mon, tmp_time.tm_mday, tmp_time.tm_hour, tmp_time.tm_min)
    if not os.path.exists(runs_folder):
        os.mkdir(runs_folder)
    if not os.path.exists(f"{runs_folder}/modules"):
        os.mkdir(f"{runs_folder}/modules")
    super_dict = {
        "lr": 65e-6,
        "batch_size": 8,
        "Molucula_InputChannel": 77,
        "output_channel": 8192,
        "Protein_InputChannel": 19,
        "p": 0.1,
        "pad_size": (1200,300),
        "device": cuda,
        "mean": 0,
        "step_size": 1e5,
        "gamma": 1.1,
        "kernel_size": (4,3),
        "loss_type": "L1",
        "train_cutoff": 0.8,
        "number": 4,
        "valid_cutoff": 0.5,
        "database_name": "pdbbind",
        "run_folder": runs_folder,
        "optimizer": "Adam",
    }
    with open(f"{runs_folder}/modules/super_dict.pickle", 'wb') as file:
        pickle.dump(super_dict, file)
    module = Modules(super_dict)
    # initialize_model(module)
    module = module.to(cuda)
    # module_complied = torch.compile(module, mode="max-autotune")
    trainer = Trainer(module, super_dict)
    # trainer = Trainer(module_complied, super_dict)

    for i in range(train_cucle):
        train_df,valid_df,test_df = train_pickles.return_all()
        train_dataset = Df_To_Dataset(train_df, 0)
        train_dataloader = DataLoader(train_dataset)
        valid_dataset = Df_To_Dataset(valid_df, 0)
        valid_dataloader = DataLoader(valid_dataset)
        match i%3:
            case 0:
                trainer.loss_leary = trainer.L1
            case 1:
                trainer.loss_leary = trainer.mse
            case 2:
                trainer.loss_leary = trainer.huber
        try:
            trainer.train(train_dataloader, i, valid_dataloader)
            if i % log_batch == 0:
                module_save_path = f"{runs_folder}/modules/{trainer.title}_{i}.pth"
                torch.save(module, module_save_path)
            vailder = Vail_When_Train(module, super_dict)
            rmse,mae, pearson_corr = vailder.vaild_All(valid_dataloader)
            with open(f"{runs_folder}/MAE.txt","a+") as log:
                log.write(f"{i}__{rmse}__{mae}__{pearson_corr}\n")
        except KeyboardInterrupt:
            module_save_path = f"{runs_folder}/modules/{trainer.title}_{i}.pth"
            torch.save(module, module_save_path)
            continue
        except Jumpout:
            break
        except Exception as e:
            module_save_path = f"{runs_folder}/modules/{trainer.title}_{i}.pth"
            torch.save(module, module_save_path)
            with open(f"{runs_folder}/Err.txt","a+") as log:
                log.write(f"{i}__{e}\n")
            exit()
            #continue

    module_save_path = f"{runs_folder}/modules/{trainer.title}_all.pth"
    torch.save(module, module_save_path)

    test_pickles = Read_Pick(folder, test_file_name)
    valid_df, test_df = test_pickles.return_vt_part()
    valid_dataset = Df_To_Dataset(valid_df, 0)
    valid_dataloader = DataLoader(valid_dataset)
    test_dataset = Df_To_Dataset(test_df, 0)
    test_dataloder = DataLoader(test_dataset)

    tester = Tester(module, cuda, trainer.time_log, runs_folder)
    tester.test(test_dataloder, f"""{test_file_name}_{super_dict["database_name"]}""")
