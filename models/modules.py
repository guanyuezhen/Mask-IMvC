import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act=nn.ReLU()):
        super(GCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.act = act

        self.encoder_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims)):
            if i == 0:
                self.encoder_layers.append(pyg_nn.GCNConv(self.input_dim, self.hidden_dims[i]))
            else:
                self.encoder_layers.append(pyg_nn.GCNConv(self.hidden_dims[i - 1], self.hidden_dims[i]))
        self.encoder_layers.append(pyg_nn.GCNConv(self.hidden_dims[-1], self.latent_dim))

    def forward(self, features, graph):
        graph.fill_diagonal_(1)
        degree = graph.sum(dim=1)
        degree_inv_sqrt = torch.diag(degree.pow(-0.5))
        graph = degree_inv_sqrt @ graph @ degree_inv_sqrt
        edge_index = graph.nonzero(as_tuple=True)
        edge_index = torch.stack(edge_index).to(torch.long)
        edge_weight = graph[edge_index[0], edge_index[1]]
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight)
        z, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for _, layer in enumerate(self.encoder_layers):
            z = layer(z, edge_index, edge_weight=edge_attr)
            z = self.act(z)

        return z


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act=nn.ReLU()):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.act = act

        encoder_layers = []
        for i in range(len(self.hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(self.input_dim, self.hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
            encoder_layers.append(self.act)
        encoder_layers.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))
        encoder_layers.append(self.act)
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        z = self.encoder(x)

        return z


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act=nn.ReLU()):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(reversed(hidden_dims))
        self.latent_dim = latent_dim
        self.act = act

        decoder_layers = []

        for i in range(len(self.hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(self.latent_dim, self.hidden_dims[i]))
            else:
                decoder_layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
            decoder_layers.append(self.act)
        decoder_layers.append(nn.Linear(self.hidden_dims[-1], self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x_rec = self.decoder(x)

        return x_rec


class FusionModule(nn.Module):
    def __init__(self, latent_dim, view_num, embedding_dim, fusion_mode="average_sum"):
        super(FusionModule, self).__init__()
        self.latent_dim = latent_dim
        self.view_num = view_num
        self.embedding_dim = embedding_dim
        self.fusion_mode = fusion_mode
        if fusion_mode == "concat":
            self.embedding_layer = nn.Linear(latent_dim * view_num, embedding_dim)
        elif fusion_mode == "weighted_sum":
            self.weight_assignment = nn.Sequential(
                nn.Linear(latent_dim * view_num, view_num)
            )
            self.embedding_layer = nn.Linear(latent_dim, embedding_dim)
        else:
            self.embedding_layer = nn.Linear(latent_dim, embedding_dim)

    def forward(self, specific_x_, observed_indices, sample_mask):
        specific_x = []
        for v in range(self.view_num):
            specific_x.append(observed_indices[v] @ specific_x_[v])
        if self.fusion_mode == "concat":
            fusion_x = torch.cat(specific_x, dim=1)
            joint_x = self.embedding_layer(fusion_x)
        elif self.fusion_mode == "weighted_sum":
            weights = self.weight_assignment(torch.cat(specific_x, dim=1))
            weights = torch.softmax(weights * sample_mask, dim=-1)
            weights_chunk = torch.chunk(weights, self.view_num, dim=1)
            fusion_x = []
            for v in range(self.view_num):
                fusion_x.append(specific_x[v] * weights_chunk[v])
            fusion_x = sum(fusion_x)
            joint_x = self.embedding_layer(fusion_x)
        elif self.fusion_mode == "average_sum":
            sample_mask = sample_mask.unsqueeze(1).repeat(1, self.latent_dim, 1)
            x_stack = torch.stack(specific_x, dim=-1)  # (n, d, v)
            weighted_sum = torch.sum(sample_mask * x_stack, dim=-1)
            weights_sum = sample_mask.sum(dim=-1)
            fusion_x = weighted_sum / weights_sum
            joint_x = self.embedding_layer(fusion_x)
        else:
            joint_x = None

        return joint_x
