import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
import torch_geometric.transforms as Trans
from torch_geometric.nn import GCNConv, ChebConv, GATv2Conv
from torch_geometric.nn import Linear as pyg_Linear
from torch_geometric.utils import *
import math
from tensorboardX import SummaryWriter
import time

class Jumpout(Exception):
    pass

class MoluculaCNN(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(MoluculaCNN, self).__init__()
        self.conv1 = GCNConv(in_channels=input_channel,
                             out_channels=16 * input_channel)
        self.conv2 = GCNConv(
            in_channels=16 * input_channel, out_channels=output_channel
        )
        self.conv3 = GCNConv(
            in_channels=output_channel, out_channels=8 * output_channel
        )
        self.conv4 = GCNConv(
            in_channels=8 * output_channel, out_channels=output_channel
        )

    def simple_con1(self, graph):
        x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        tmp_Graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        return tmp_Graph

    def CNN1(self, graph):
        x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_attr
        edge_index = edge_index.type(torch.long)
        tmp_Graph = self.simple_con1(
            Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        )
        x = tmp_Graph.x

#        Feature_relu = F.elu(x)
        Feature_relu = x

        Feature_cov2 = self.conv2(x=Feature_relu, edge_index=edge_index)
        tmp_Graph = Data(x=Feature_cov2, edge_index=edge_index)
        return tmp_Graph

    def CNN2(self, graph):
        x, edge_index = graph.x, graph.edge_index
        edge_index = edge_index.type(torch.long)
        x = self.conv3(x=x, edge_index=edge_index)

#        Feature_relu = F.elu(x)
        Feature_relu = x

        Feature_cov2 = self.conv4(x=Feature_relu, edge_index=edge_index)
        tmp_Graph = Data(x=Feature_cov2, edge_index=edge_index)
        return tmp_Graph

    def CNN3(self, graph):
        x, edge_index = graph.x, graph.edge_index
        edge_index = edge_index.type(torch.long)
        x = self.conv3(x=x, edge_index=edge_index)

#        Feature_relu = F.elu(x)
        Feature_relu = x

        Feature_cov2 = self.conv4(x=Feature_relu, edge_index=edge_index)
        tmp_Graph = Data(x=Feature_cov2, edge_index=edge_index)
        return tmp_Graph

    def CNN4(self, graph):
        x, edge_index = graph.x, graph.edge_index
        edge_index = edge_index.type(torch.long)
        x = self.conv3(x=x, edge_index=edge_index)

#        Feature_relu = F.elu(x)
        Feature_relu = x

        Feature_cov2 = self.conv4(x=Feature_relu, edge_index=edge_index)
        tmp_Graph = Data(x=Feature_cov2, edge_index=edge_index)
        return tmp_Graph

    def forward(self, graph):
        Cnn_graph = self.CNN1(graph)
        Cnn_graph = self.CNN2(Cnn_graph)
        Cnn_graph2 = self.CNN3(Cnn_graph)
        Cnn_graph3 = self.CNN4(Cnn_graph2)
        output_graph = Data(
            x=F.elu(sum(sum(Cnn_graph.x, Cnn_graph2.x), Cnn_graph3.x)),
            edge_index=Cnn_graph.edge_index,
        )
        # return Cnn_graph
        return output_graph

class Modules(nn.Module):
    def __init__(
        self,
        super_dict,
    ):
        super().__init__()

        def return_pooled_size(input_channels,kernel_size,stride):
            h_in = input_channels[0]
            w_in = input_channels[1]
            h_out = math.ceil((h_in - (kernel_size[0]-1))/stride[0])
            w_out = math.ceil((w_in - (kernel_size[1]-1))/stride[1])
            return (h_out,w_out)

        Molucula_InputChannel = super_dict["Molucula_InputChannel"]
        output_channel = super_dict["output_channel"]
        p = super_dict["p"]
        pad_size = super_dict["pad_size"]
        device = super_dict["device"]
        kernel_size = super_dict["kernel_size"]
        output_size = super_dict["output_size"]

        self.Molucula = MoluculaCNN(
            input_channel=Molucula_InputChannel, output_channel=output_channel)

        self.Chem_bias = torch.randn(size=[output_channel,output_channel], requires_grad=True,device=device)

        self.dropout = nn.Dropout(p=p)

        self.output_pad_size_0 = pad_size[0]

        self.paded_liner1 = nn.Linear(pad_size[0], output_channel)

        self.cov2d1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=kernel_size)
        
        self.conv_out_size = (
            output_channel-1*(kernel_size[0])+1, output_channel-1*(kernel_size[1])+1)
        
        self.cov2d2 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=kernel_size)
        
        self.conv_out_size = (
            self.conv_out_size[0]-1*(kernel_size[0])+1, self.conv_out_size[1]-1*(kernel_size[1])+1)
        
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size,stride=(int(kernel_size[0]/2),int(kernel_size[1]/2)))

        self.conv_out_size = return_pooled_size(self.conv_out_size,kernel_size,(int(kernel_size[0]/2),int(kernel_size[1]/2)))
        self.cov2d3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=kernel_size)
        
        self.conv_out_size = (
            self.conv_out_size[0]-1*(kernel_size[0])+1, self.conv_out_size[1]-1*(kernel_size[1])+1)
        
        self.cov2d4 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size)
        
        self.conv_out_size = (
            self.conv_out_size[0]-1*(kernel_size[0])+1, self.conv_out_size[1]-1*(kernel_size[1])+1)

        self.liner = nn.Linear(
            in_features=self.conv_out_size[0], out_features=output_size)
        self.liner2 = nn.Linear(
            in_features=self.conv_out_size[1], out_features=1)
       

    def forward(self, data):
        Molucula_graph = data[0]

        Molucula_Output = self.Molucula.forward(Molucula_graph)

        Molucula_Output_adj = to_dense_adj(Molucula_Output.edge_index.type(torch.long))[
            0
        ]
        Molucula_Output_Add = torch.mm(
            Molucula_Output_adj, Molucula_Output.x)
        #Molucula_Output_bias = torch.mm(Molucula_Output_Add, self.Chem_bias)
        #Output_x = F.elu(Molucula_Output_bias)
        Output_x = F.elu(Molucula_Output_Add)

        Output_x = self.dropout(Output_x)
       
        

        top_pad = int(0.5 * (self.output_pad_size_0 - Output_x.size(0)))
        down_pad = int(self.output_pad_size_0 - Output_x.size(0) - top_pad)
        Output_x = F.pad(Output_x, (0, 0, top_pad, down_pad))
        _ = sum(Output_x)

        tmp_output = Output_x.T
        tmp_output = self.paded_liner1(tmp_output)
        tmp_output = tmp_output.T

        tmp_output = torch.reshape(
            tmp_output, (1, 1, tmp_output.shape[0], tmp_output.shape[1]))

        output = self.cov2d1(tmp_output)
        output = self.cov2d2(output)
        output = self.pooling(output)
        output = self.cov2d3(output)
        output = self.cov2d4(output)

        # output = F.elu(output)

        output = torch.reshape(
            output, (output.shape[1], output.shape[2], output.shape[3]))

        output = self.liner2(output)
        output = torch.reshape(output, (output.shape[2], output.shape[1]))
        output = self.liner(output)

        output = torch.reshape(output, (output.shape[1],))
        return output


class Trainer(object):
    def __init__(self, model, super_dict):
        self.model = model
        self.super_dict = super_dict
        self.super_string = f"""database_{super_dict["database_name"]}lr_{super_dict["lr"]}__batch_size_{super_dict["batch_size"]}__output_channel_{super_dict["output_channel"]}__pad_size_{super_dict["pad_size"]}__mean_{super_dict["mean"]}__kernel_size_{super_dict["kernel_size"]}__p_{super_dict["p"]}__{super_dict["loss_type"]}__{super_dict["optimizer"]}"""
        self.batch_size = super_dict["batch_size"]
        lr = super_dict["lr"]
        self.train_cutoff = super_dict["train_cutoff"]
        self.valid_cutoff = super_dict["valid_cutoff"]
        self.number = super_dict["number"]

        match super_dict["optimizer"]:
            case "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            case "Adadelta":
                self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        self.optimizer.zero_grad()

        self.huber = nn.HuberLoss()
        self.mse = nn.MSELoss()
        self.L1 = nn.L1Loss()
        match super_dict["loss_type"]:
            case "huber":
                self.loss_leary = self.huber
            case "mse":
                self.loss_leary = self.mse
            case "L1":
                self.loss_leary = self.L1

        tmp_time = time.localtime(time.time())
        self.time_log = "{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}__".format(
            tmp_time.tm_mon, tmp_time.tm_mday, tmp_time.tm_hour, tmp_time.tm_min
        )
        self.title = f"{self.time_log}_{self.super_string}"
        self.runs_folder = super_dict["run_folder"]
        step_size = super_dict["step_size"]
        gamma = super_dict["gamma"]
        self.scheduler = StepLR(
            self.optimizer, step_size=step_size, gamma=gamma, verbose=True
        )

    def train(self, train_loader, cycle, valid_dataloader):

        length = len(train_loader)

        device = self.super_dict["device"]
        self.writer = SummaryWriter(
            f"{self.runs_folder}/{self.time_log}log_lr_{self.super_string}__cycle_{cycle}"
        )
        
        
        batch_size = self.super_dict["batch_size"]
        # loss_sum = 0
        n = 1
        pre_list = []
        label_list = []
        jump_label = 0
        torch.cuda.empty_cache()
        for data_sampler in train_loader:
            torch.cuda.empty_cache()
            chem_graph, label = data_sampler

            chem_graph.edge_index = chem_graph.edge_index.to(torch.long)
            
            chem_graph = chem_graph.to(device)
            label = label.to(device)
            
            data = [chem_graph]
            try:
                pre = self.model.forward(data)
                pre_list.append(pre)
                label_list.append(label)
            except Exception as e:
                print(e)
                continue
            loss = self.loss_leary(pre, label)
            print(
                f"pre-->{pre.tolist()}_label-->{label.tolist()}_loss-->{loss.tolist()}"
            )
            print(math.sqrt(F.mse_loss(pre, label)))
            self.writer.add_scalar("loss", float(loss), global_step=n)
            self.writer.add_scalar("mse", float(
                F.mse_loss(pre, label)), global_step=n)
            self.writer.add_scalar("rmse", float(
                math.sqrt(F.mse_loss(pre, label))), global_step=n)
            self.writer.add_scalar(
                "huber", float(F.huber_loss(pre, label)), global_step=n
            )
            self.writer.add_scalar("pre", float(pre), global_step=n)
            loss.backward()
            #scheduler.step()

            if (n + 1) % batch_size == 0:
                print("update")
                processed = 100*(n/length)
                print(f"processed-->{processed}")
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                mse = F.mse_loss(torch.tensor(pre_list),
                                 torch.tensor(label_list))
                self.writer.add_scalar("batch_mse", float(mse), global_step=n)
                self.writer.add_scalar("batch_rmse", float(
                    math.sqrt(mse)), global_step=n)
                pre_list = []
                label_list = []
                if float(math.sqrt(mse)) < self.train_cutoff:
                    jump_label += 1
                else:
                    jump_label = 0
                if jump_label > self.number:
                    sec_model = self.model
                    with torch.no_grad():
                        tmp_tester = Tester(
                            sec_model, self.super_dict["device"], self.time_log,self.runs_folder)
                        mse = tmp_tester.test(
                            valid_dataloader, f"""{self.super_dict["database_name"]}_valid_cycle_{cycle}_number_{n}""")
                        if math.sqrt(mse) < self.valid_cutoff:
                            raise Jumpout
                        else:
                            jump_label = 0
                            continue

            # loss_sum = loss_sum + loss
            n = n + 1
            del chem_graph
            del data
            del label
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

    def train_static_lr(self, train_loader, cycle, valid_dataloader):
        device = self.super_dict["device"]
        length = len(train_loader)
        self.writer = SummaryWriter(
            f"{self.runs_folder}/{self.time_log}static_lr__{self.super_string}__cycle_{cycle}"
        )
        batch_size = self.super_dict["batch_size"]
        # loss_sum = 0
        n = 1
        pre_list = []
        label_list = []
        jump_label = 0
        torch.cuda.empty_cache()
        for data_sampler in train_loader:
            torch.cuda.empty_cache()
            chem_graph, label = data_sampler
            chem_graph.edge_index = chem_graph.edge_index.to(torch.long)
            chem_graph = chem_graph.to(device)
            label = label.to(device)
            data = [chem_graph]
            try:
                pre = self.model.forward(data)
                pre_list.append(pre)
                label_list.append(label)
            except Exception as e:
                print(e)
                continue
            loss = self.loss_leary(pre, label)
            print(
                f"pre-->{pre.tolist()}_label-->{label.tolist()}_loss-->{loss.tolist()}"
            )
            print(math.sqrt(F.mse_loss(pre, label)))
            self.writer.add_scalar("loss", float(loss), global_step=n)
            self.writer.add_scalar("mse", float(
                F.mse_loss(pre, label)), global_step=n)
            self.writer.add_scalar("rmse", float(
                math.sqrt(F.mse_loss(pre, label))), global_step=n)
            self.writer.add_scalar(
                "huber", float(F.huber_loss(pre, label)), global_step=n
            )
            #self.writer.add_scalar("pre", float(pre), global_step=n)
            loss.backward()

            if (n + 1) % batch_size == 0:
                print("update")
                processed = 100*(n/length)
                print(f"processed-->{processed}")
                self.optimizer.step()
                self.optimizer.zero_grad()
                pre_tensor = torch.cat(pre_list)
                label_tensor = torch.cat(label_list)
                mse = F.mse_loss(pre_tensor, label_tensor)
                self.writer.add_scalar("batch_mse", float(mse), global_step=n)
                self.writer.add_scalar("batch_rmse", float(
                    math.sqrt(mse)), global_step=n)
                pre_list = []
                label_list = []
                if float(math.sqrt(mse)) < self.train_cutoff:
                    jump_label += 1
                else:
                    jump_label = 0
                if jump_label > self.number:
                    sec_model = self.model
                    with torch.no_grad():
                        tmp_tester = Tester(
                            sec_model, self.super_dict["device"], self.time_log,self.runs_folder)
                        mse = tmp_tester.test(
                            valid_dataloader, f"""{self.super_dict["database_name"]}_valid_cycle_{cycle}_number_{n}""")
                        if math.sqrt(mse) < self.valid_cutoff:
                            raise Jumpout
                        else:
                            jump_label = 0
                            continue

            # loss_sum = loss_sum + loss
            n = n + 1
            del chem_graph
            del data
            del label
        self.optimizer.step()
        self.optimizer.zero_grad()
        # return loss_sum / n

    def train_no_address(self, train_loader, cycle, valid_dataloader):

        length = len(train_loader)

        device = self.super_dict["device"]
        self.writer = SummaryWriter(
            f"{self.runs_folder}/{self.time_log}log_lr_{self.super_string}__cycle_{cycle}"
        )
        step_size = self.super_dict["step_size"]
        gamma = self.super_dict["gamma"]
        scheduler = StepLR(
            self.optimizer, step_size=step_size, gamma=gamma, verbose=True
        )
        batch_size = self.super_dict["batch_size"]
        # loss_sum = 0
        n = 1
        pre_list = []
        label_list = []
        jump_label = 0
        torch.cuda.empty_cache()
        for data_sampler in train_loader:
            torch.cuda.empty_cache()
            proten_graph, chem_graph, label, sequence = data_sampler
            proten_graph.edge_index = proten_graph.edge_index.to(torch.long)
            chem_graph.edge_index = chem_graph.edge_index.to(torch.long)
            proten_graph = proten_graph.to(device)
            chem_graph = chem_graph.to(device)
            label = label.to(device)
            sequence = sequence.to(device)
            data = [proten_graph, chem_graph, sequence]
            try:
                pre = self.model.forward_no_softmax(data)
                pre_list.append(pre)
                label_list.append(label)
            except Exception as e:
                print(e)
                continue
            loss = self.loss_leary(pre, label)
            print(
                f"pre-->{pre.tolist()}_label-->{label.tolist()}_loss-->{loss.tolist()}"
            )
            print(math.sqrt(F.mse_loss(pre, label)))
            self.writer.add_scalar("loss", float(loss), global_step=n)
            self.writer.add_scalar("mse", float(
                F.mse_loss(pre, label)), global_step=n)
            self.writer.add_scalar("rmse", float(
                math.sqrt(F.mse_loss(pre, label))), global_step=n)
            self.writer.add_scalar(
                "huber", float(F.huber_loss(pre, label)), global_step=n
            )
            self.writer.add_scalar("pre", float(pre), global_step=n)
            loss.backward()
            scheduler.step()

            if (n + 1) % batch_size == 0:
                print("update")
                print(100*(n/length))
                self.optimizer.step()
                self.optimizer.zero_grad()
                mse = F.mse_loss(torch.tensor(pre_list),
                                 torch.tensor(label_list))
                self.writer.add_scalar("batch_mse", float(mse), global_step=n)
                self.writer.add_scalar("batch_rmse", float(
                    math.sqrt(mse)), global_step=n)
                pre_list = []
                label_list = []
                if float(math.sqrt(mse)) < self.train_cutoff:
                    jump_label += 1
                else:
                    jump_label = 0
                if jump_label > self.number:
                    sec_model = self.model
                    with torch.no_grad():
                        tmp_tester = Tester(
                            sec_model, self.super_dict["device"], self.time_log,self.runs_folder)
                        mse = tmp_tester.test(
                            valid_dataloader, f"""{self.super_dict["database_name"]}_valid_cycle_{cycle}_number_{n}""")
                        if math.sqrt(mse) < self.valid_cutoff:
                            raise Jumpout
                        else:
                            jump_label = 0
                            continue

            # loss_sum = loss_sum + loss
            n = n + 1
            del proten_graph
            del chem_graph
            del data
            del label
        self.optimizer.step()
        self.optimizer.zero_grad()
        scheduler.step()


class Tester(object):
    def __init__(self, model, device, times,runs_folder):
        self.model = model
        self.device = device
        self.model.eval()
        self.pre_list = []
        self.label_list = []
        self.times = times
        self.runs_folder = runs_folder

    def test(self, test_loader, file_name):
        self.loger = SummaryWriter(
            f"{self.runs_folder}/vt_loger_{self.times}_file_{file_name}")
        device = self.device
        with torch.no_grad():
            for index, data in enumerate(test_loader):
                chem_graph, label= data
                chem_graph = chem_graph.to(device)
                label = label.to(device)
                data = [chem_graph]
                try:
                    pre = self.model.forward(data)
                except:
                    continue
                self.pre_list.append(pre)
                self.label_list.append(label)
                mse = F.mse_loss(torch.tensor(self.pre_list),
                                 torch.tensor(self.label_list))
                print(f"{index}_{mse}")
                self.loger.add_scalar("mse", mse, global_step=index)
                self.loger.add_scalar(
                    "rmse", math.sqrt(mse), global_step=index)
        return mse


class Prediction(object):
    def __init__(self, model):
        self.model = model

    def pre(self, Graph_List):
        pre = self.model.forward(Graph_List)
        return pre
