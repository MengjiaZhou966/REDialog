import torch.nn as nn
import torch
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):

        input_dot = self.input_linear(input) #nan: cal the weight for the same word
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None,0])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)

        # return torch.cat([input, output_one], dim=-1)
        return weight_one

class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers, dropout, self_loop = False):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = dropout

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()
        self.self_loop = self_loop


    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            if self.self_loop:
                AxW = AxW  + self.weight_list[l](outputs)  # self loop
            else:
                AxW = AxW

            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out

class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on multihead attention """

    def __init__(self,mem_dim, layers, heads, dropout):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = dropout
        # self.update_adj = nn.ModuleList()

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                # self.update_adj.append(SelfAttention(self.mem_dim + self.head_dim * (j+1)))
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):
        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[:,:,:,i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))
                # adj = self.update_adj[index](outputs, outputs, adj_list[:,:,:,i])

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)
        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)
        return out
