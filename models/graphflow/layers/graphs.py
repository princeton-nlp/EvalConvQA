'''
Created on Nov, 2018

@author: hugo

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.generic_utils import to_cuda, get_range_vector, get_sinusoid_encoding_table
from ..layers.common import *
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", Warning)

INF = 1e20
VERY_SMALL_NUMBER = 1e-12
COMBINE_RATIO = 0.9

class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, topk, epsilon, n_spatial_kernels, use_spatial_kernels=True, \
            use_position_enc=False, position_emb_size=10, max_position_distance=160, num_pers=1, device=None):
        super(GraphLearner, self).__init__()
        self.device = device
        self.topk = topk
        self.epsilon = epsilon
        self.use_spatial_kernels = use_spatial_kernels
        self.use_position_enc = use_position_enc
        self.max_position_distance = max_position_distance
        # self.linear_sim = nn.Linear(input_size, hidden_size, bias=False)

        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        print('[ Multi-perspective GraphLearner: {} ]'.format(num_pers))


        if use_spatial_kernels:
            print('[ Using spatial Gaussian kernels ]')
            if use_position_enc:
                print('[ Using sinusoid position encoding ]')
                # Position encoding
                self.position_enc = nn.Embedding.from_pretrained(
                    get_sinusoid_encoding_table(self.max_position_distance + 1, position_emb_size, padding_idx=0, device=device),
                    freeze=True)

                # Parameters of the Gaussian kernels
                self.mean_dis = nn.Parameter(torch.Tensor(n_spatial_kernels, position_emb_size))
                self.mean_dis.data.uniform_(-1, 1)
                self.precision_inv_dis = nn.Parameter(torch.Tensor(n_spatial_kernels, position_emb_size))
                self.precision_inv_dis.data.uniform_(0.0, 1.0)
            else:
                # Parameters of the Gaussian kernels
                self.mean_dis = nn.Parameter(torch.Tensor(n_spatial_kernels, 1))
                self.mean_dis.data.uniform_(0, 1)
                self.precision_inv_dis = nn.Parameter(torch.Tensor(n_spatial_kernels, 1))
                self.precision_inv_dis.data.uniform_(0.0, 1.0)

    def forward(self, context, ctx_mask):
        """
        Parameters
        :context, (batch_size, turn_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :adjacency_matrix, (batch_size, turn_size, ctx_size, ctx_size)
        """
        markoff_value = -INF

        # 1)
        # context_fc = torch.relu(self.linear_sim(context))
        # attention = torch.matmul(context_fc, context_fc.transpose(-1, -2))

        # # 2)
        # context_fc = context.unsqueeze(2) * self.weight_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
        # attention = torch.mean(torch.matmul(context_fc, context_fc.transpose(-1, -2)), dim=2)


        # 3) Best attention mechanism
        context_fc = context.unsqueeze(2) * torch.relu(self.weight_tensor).unsqueeze(0).unsqueeze(0).unsqueeze(-2)
        attention = torch.mean(torch.matmul(context_fc, context.unsqueeze(2).transpose(-1, -2)), dim=2)


        # # 4）weighted cosine
        # context_fc = context.unsqueeze(2) * self.weight_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
        # context_norm = F.normalize(context_fc, p=2, dim=-1)
        # attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(2)
        # markoff_value = 0


        if ctx_mask is not None:
            # print("ctx mask")
            mask1 = (1 - ctx_mask.byte().unsqueeze(1).unsqueeze(-1)).to(torch.bool)
            # print(mask1.dtype)
            attention = attention.masked_fill_(mask1, markoff_value)
            mask2 = (1 - ctx_mask.byte().unsqueeze(1).unsqueeze(-2)).to(torch.bool)
            # print(mask2.dtype)
            attention = attention.masked_fill_(mask2, markoff_value)

        if self.use_spatial_kernels:
            # shape: (batch_size, turn_size, n_spatial_kernels, ctx_size, ctx_size)
            spatial_attention = self.get_spatial_attention(attention.shape[:3])
            # joint_attention = COMBINE_RATIO * torch.softmax(attention, dim=-1).unsqueeze(2) + (1 - COMBINE_RATIO) * spatial_attention / torch.sum(spatial_attention, dim=-1, keepdim=True)
            weighted_adjacency_matrix = self.build_knn_neighbourhood(attention, self.topk, attention, spatial_attention)
        else:
            if self.topk is not None:
                weighted_adjacency_matrix = self.build_knn_neighbourhood(attention, self.topk)

            if self.epsilon is not None:
                weighted_adjacency_matrix = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        return weighted_adjacency_matrix


    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix


    def build_knn_neighbourhood(self, attention, topk, semantic_attention=None, spatial_attention=None, markoff_value=-INF):
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        if self.use_spatial_kernels:
            # semantic_attention = semantic_attention.unsqueeze(2).expand(-1, -1, spatial_attention.size(2), -1, -1)
            semantic_attn_chosen = torch.gather(semantic_attention, dim=-1, index=knn_ind)
            semantic_attn_chosen = torch.softmax(semantic_attn_chosen, dim=-1)

            expand_knn_ind = knn_ind.unsqueeze(2).expand(-1, -1, spatial_attention.size(2), -1, -1)
            spatial_attn_chosen = torch.gather(spatial_attention, dim=-1, index=expand_knn_ind)
            spatial_attn_chosen = spatial_attn_chosen / torch.sum(spatial_attn_chosen, dim=-1, keepdim=True)

            attn_chosen = semantic_attn_chosen.unsqueeze(2) * spatial_attn_chosen
            weighted_adjacency_matrix = to_cuda(torch.zeros_like(spatial_attention).scatter_(-1, expand_knn_ind, attn_chosen), self.device)
        else:
            weighted_adjacency_matrix = to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix

    def get_spatial_attention(self, shape):
        # Compute pseudo-coordinates for context words
        batch_size, turn_size, ctx_size = shape
        ctx_token_idx = get_range_vector(ctx_size, self.device)
        pseudo_coord = ctx_token_idx.unsqueeze(-1) - ctx_token_idx.unsqueeze(0)
        if self.use_position_enc:
            # Truncate
            pseudo_coord = torch.clamp(torch.abs(pseudo_coord) + 1, max=self.max_position_distance)
            pseudo_coord = self.position_enc(pseudo_coord)
            # Use Gaussian kernel to model attention over distance
            spatial_attention = self.get_multivariate_gaussian_weights(pseudo_coord)
        else:
            # Truncate & scale
            # pseudo_coord = torch.clamp(pseudo_coord, min=-self.max_position_distance, max=self.max_position_distance)
            pseudo_coord = torch.clamp(torch.abs(pseudo_coord.float()), max=self.max_position_distance) / self.max_position_distance
            # Use Gaussian kernel to model attention over distance
            spatial_attention = self.get_gaussian_weights(pseudo_coord)

        # shape: (batch_size, turn_size, n_spatial_kernels, ctx_size, ctx_size)
        spatial_attention = spatial_attention.unsqueeze(0).unsqueeze(0).expand(batch_size, turn_size, -1, -1, -1)
        return spatial_attention

    def get_gaussian_weights(self, pseudo_coord):
        '''
        ## Inputs:
        - pseudo_coord (ctx_size, ctx_size)
        ## Returns:
        - weights (n_spatial_kernels, ctx_size, ctx_size)
        '''
        # compute weights
        diff = (pseudo_coord.view(1, -1) - self.mean_dis)**2
        weights = torch.exp(-0.5 * diff * (self.precision_inv_dis**2))

        # shape: (n_spatial_kernels, ctx_size, ctx_size)
        weights = weights.view((-1,) + pseudo_coord.shape)
        return weights

    def get_multivariate_gaussian_weights(self, pseudo_coord):
        '''
        ## Inputs:
        - pseudo_coord (ctx_size, ctx_size, dim)
        ## Returns:
        - weights (n_spatial_kernels, ctx_size, ctx_size)
        '''
        # compute weights
        diff = (pseudo_coord.view(1, -1, pseudo_coord.size(-1)) - self.mean_dis.view(-1, 1, self.mean_dis.size(-1)))**2
        weights = torch.exp(-0.5 * torch.sum(diff * (self.precision_inv_dis.unsqueeze(1))**2, dim=-1))

        # shape: (n_spatial_kernels, ctx_size, ctx_size)
        weights = weights.view((-1,) + pseudo_coord.shape[:2])
        return weights

class ContextGraphNN(nn.Module):
    def __init__(self, hidden_size, n_spatial_kernels, use_spatial_kernels=True, graph_hops=1, bignn=False, device=None):
        super(ContextGraphNN, self).__init__()
        print('[ Using {}-hop ContextGraphNN ]'.format(graph_hops))
        self.graph_hops = graph_hops
        self.use_spatial_kernels = use_spatial_kernels
        if self.use_spatial_kernels:
            self.linear_kernels = nn.ModuleList([nn.Linear(hidden_size, hidden_size // n_spatial_kernels, bias=False) for _ in range(n_spatial_kernels)])
        else:
            n_spatial_kernels = 1
        self.gru_step = GRUStep(hidden_size, hidden_size // n_spatial_kernels * n_spatial_kernels)
        if bignn:
            self.gated_fusion = GatedFusion(hidden_size)
            self.update = self.bignn_update
        else:
            self.update = self.gnn_update

        print('[ Using graph type: dynamic ]')


    def forward(self, node_state, weighted_adjacency_matrix):
        node_state = self.update(node_state, weighted_adjacency_matrix)
        return node_state

    def bignn_update(self, node_state, weighted_adjacency_matrix):
        weighted_adjacency_matrix_in = torch.softmax(weighted_adjacency_matrix, dim=-1)
        weighted_adjacency_matrix_out = torch.softmax(weighted_adjacency_matrix.transpose(-1, -2), dim=-1)

        for _ in range(self.graph_hops):
            agg_state_in = self.aggregate_avgpool(node_state, weighted_adjacency_matrix_in)
            agg_state_out = self.aggregate_avgpool(node_state, weighted_adjacency_matrix_out)
            agg_state = self.gated_fusion(agg_state_in, agg_state_out)
            node_state = self.gru_step(node_state, agg_state)
        return node_state

    def gnn_update(self, node_state, weighted_adjacency_matrix):
        weighted_adjacency_matrix = torch.softmax(weighted_adjacency_matrix, dim=-1)


        for _ in range(self.graph_hops):
            agg_state = self.aggregate_avgpool(node_state, weighted_adjacency_matrix)
            node_state = self.gru_step(node_state, agg_state)
        return node_state

    def aggregate_avgpool(self, node_state, weighted_adjacency_matrix):
        # Information aggregation
        if self.use_spatial_kernels:
            # Joint aggregation
            agg_state = torch.cat([self.linear_kernels[i](torch.matmul(weighted_adjacency_matrix[:, i], node_state)) for i in range(weighted_adjacency_matrix.size(1))], -1)
        else:
            agg_state = torch.matmul(weighted_adjacency_matrix, node_state)
        return agg_state


# Static GNN
class StaticContextGraphNN(nn.Module):
    def __init__(self, hidden_size, graph_hops=1, device=None):
        super(StaticContextGraphNN, self).__init__()
        print('[ Using {}-hop GraphNN ]'.format(graph_hops))
        self.device = device
        self.graph_hops = graph_hops
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False)

        # Static graph
        self.static_graph_mp = GraphMessagePassing()
        self.static_gated_fusion = GatedFusion(hidden_size)
        self.static_gru_step = GRUStep(hidden_size, hidden_size)

        print('[ Using graph type: static ]')

    def forward(self, node_state, adj):
        '''Static graph update'''
        node2edge, edge2node = adj

        # Shape: (batch_size, num_edges, num_nodes)
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)

        for _ in range(self.graph_hops):
            bw_agg_state = self.static_graph_mp(node_state, node2edge, edge2node)
            fw_agg_state = self.static_graph_mp(node_state, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            agg_state = self.static_gated_fusion(fw_agg_state, bw_agg_state)
            node_state = self.static_gru_step(node_state, agg_state)
        return node_state


class GraphMessagePassing(nn.Module):
    def __init__(self):
        super(GraphMessagePassing, self).__init__()

    def forward(self, node_state, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size

        # Add self-loop
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, node2edge_emb) + node_state) / norm_
        return agg_state
