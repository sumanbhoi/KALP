import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import dill
from dnc import DNC
import math

'''
Our model
'''

class GraphAttentionV2Layer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_heads: int,
            is_concat: bool = True,
            dropout: float = 0.6,
            leaky_relu_negative_slope: float = 0.2,
            share_weights: bool = False,
            is_output_layer: bool = False,
    ) -> None:
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.is_output_layer = is_output_layer

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

        self.output_act = nn.ELU()
        self.output_dropout = nn.Dropout(dropout)

    # def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, use_einsum=True) -> torch.Tensor:
    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, use_einsum=True) -> torch.Tensor:
        """
        * `h`, "h" is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """
        # Number of nodes
        n_nodes = h.shape[0]

        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # First, calculate g_li * g_rj for all pairs of i and j
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)

        # combine g_l and g_r
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # calculate attention score e_ij
        e = self.attn(self.activation(g_sum)).squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        # print(adj_mat.shape)
        # print(n_nodes)
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        # mask e_ij based on the adjacency matrix
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # apply softmax to calculate attention
        a = self.softmax(e)
        # apply dropout
        a = self.dropout(a)

        #
        # calculate the final output for each head
        #
        if use_einsum:
            h_prime = torch.einsum('ijh,jhf->ihf', a, g_r)
        else:
            h_prime = torch.matmul(a, g_r)

        # concatenate the output of each head
        if self.is_concat:
            # print('h_prime', np.shape(h_prime))
            h_prime = h_prime.reshape(n_nodes, -1)
        else:
            h_prime = torch.mean(h_prime, dim=1)

        # do not apply activation and dropout for the output layer
        if self.is_output_layer:
            return h_prime
        # apply activation and dropout
        return self.output_dropout(self.output_act(h_prime))

class GATV2(nn.Module):
    def __init__(
            self,
            in_features: int,
            n_hidden: int,
            n_classes: int,
            adj,
            device=torch.device('cpu:0'),
            n_heads: int = 4,
            dropout: float = 0.4,
            num_of_layers: int = 2,
            share_weights: bool = True
    ) -> None:
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `num_of_layers` is the number of graph attention layers
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()
        self.num_of_layers = num_of_layers
        self.device = device
        self.x = torch.eye(in_features).to(device)
        self.adj = torch.FloatTensor(adj).to(device)
        self.layers = nn.ModuleList()

        # add input layer
        self.layers.append(GraphAttentionV2Layer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout,
                                                 share_weights=share_weights))

        # add hidden layers
        for i in range(num_of_layers - 2):
            self.layers.append(GraphAttentionV2Layer(n_hidden, n_hidden, 1, share_weights=share_weights))

        # # add output layer
        # self.layers.append(
        #     GraphAttentionV2Layer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout, share_weights=share_weights,
        #                           is_output_layer=True))

    # def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
    def forward(self):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        x = self.x
        adj = torch.reshape(self.adj,(np.shape(self.adj)[0],np.shape(self.adj)[1],1))
        print(np.shape(x))
        for i in range(self.num_of_layers-1):
            x = self.layers[i](x, adj)
            print('x size', np.shape(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        print('x', np.shape(x))
        print(np.shape(self.pe[:, :x.size(1)]))
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class KALP(nn.Module):
    def __init__(self, vocab_size, pi_adj, ni_adj, patient_sim, patient_voc, emb_dim=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=1000, dropout=0.1, device=torch.device('cpu:0')):
        super(KALP, self).__init__()
        K = len(vocab_size) - 2
        # vocab_size[0] is diag, vocab_size[1] is med, vocab_size[2] is lab test
        ''' Length here is 2 (2 types of codes)'''
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.patient_voc = patient_voc
        self.tensor_ni_adj = torch.FloatTensor(ni_adj).to(device)
        self.tensor_dco_adj = torch.FloatTensor(pi_adj).to(device)
        self.embeddings_d = nn.Embedding(vocab_size[0], emb_dim)
        self.embeddings_m = nn.Linear(vocab_size[1], emb_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.positional_encoding = nn.ModuleList(
            [PositionalEncoding(emb_dim, max_seq_length) for _ in range(K)])
        self.encoder_layers_d = nn.ModuleList(
            [EncoderLayer(emb_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder_layers_m = nn.ModuleList(
            [EncoderLayer(emb_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.sim_gat = GATV2(in_features=vocab_size[3], n_hidden=emb_dim, adj=patient_sim, n_classes=vocab_size[3],
                            device=device)
        self.pi_gat = GATV2(in_features=vocab_size[2], n_hidden=emb_dim, adj=pi_adj, n_classes=vocab_size[2], device=device)
        self.ni_gat = GATV2(in_features=vocab_size[2], n_hidden=emb_dim, adj=ni_adj, n_classes=vocab_size[2], device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Linear((2*emb_dim)+1, 1)
        self.init_weights()

    def forward(self, input):
        # input (adm, 2, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for idx, adm in enumerate(input):
            i1 = mean_embedding(self.dropout(self.embeddings_d(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            """ [1,1,64]"""

            i2 = self.dropout(self.embeddings_m(torch.FloatTensor(adm[2]).unsqueeze(dim=0).to(self.device)))
            i2 = torch.reshape(i2,(1,1,64))
            i1_seq.append(i1)
            i2_seq.append(i2)
            patient_index = self.patient_voc.word2idx[adm[4][0]]

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)


        # Transformer module

        d_embedded = self.dropout(self.positional_encoding[0](i1_seq))
        d_enc_output = d_embedded
        for enc_layer_d in self.encoder_layers_d:
            d_enc_output = enc_layer_d(d_enc_output)
        m_embedded = self.dropout(self.positional_encoding[1](i2_seq))
        m_enc_output = m_embedded
        for enc_layer_m in self.encoder_layers_m:
            m_enc_output = enc_layer_m(m_enc_output)

        #  GATv2 for patient similarity

        p_sim_matrix = self.sim_gat()
        patient_rep = p_sim_matrix[patient_index,:]

        query = d_enc_output + m_enc_output + patient_rep      # patient representation
        # print('q_before', np.shape(query))

        #  positive and negative lab impact graph encoding
        p_positive = self.pi_gat()
        q_negative = self.ni_gat()
        c = torch.unsqueeze(p_positive[0],0) - torch.unsqueeze(q_negative[0],0) * self.inter  # Lab Interaction Information
        # print('c', c)



        query = query.squeeze(dim=0)
        print('query shape', np.shape(query))
        q = query[-1:]  # (1,dim)

        if len(input) > 1:
            past_keys = query[:(query.size(0)-1)] # (seq-1, dim)
            # print('past keys shape', np.shape(past_keys))
            # past_values = np.zeros((len(input)-1, self.vocab_size[2]))
            past_values = np.zeros((len(input) - 1, 1))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                print('adm in past', np.shape(adm[3]))
                print('adm 1st val', np.shape(adm[3][0]))
                past_values[idx,:] = adm[3][0]
            past_values = torch.FloatTensor(past_values).to(self.device) # (seq-1, size)
            # print('past val', np.shape(past_values))
            # print(past_values)

        if len(input) > 1:
            # print('visit weight matrix mul before softmax', torch.mm(q, past_keys.t()))
            visit_weight = F.softmax(torch.mm(q, past_keys.t())) # (1, seq-1)
            # print('visit weight', np.shape(visit_weight))
            # print(visit_weight)
            f = visit_weight.mm(past_values) # (1, size)
            # print(np.shape(f))
            # print('f', f)
        else:
            # print(np.shape(input[0][3]))
            # print(type(input[0][3]))
            f1 = input[0][3] # We use 1st lab test value
            f = torch.FloatTensor(f1).to(self.device)
        '''R:convert O and predict'''
        output = self.output(torch.cat([q, c, f], dim=-1)) # (1, dim)
        return output[0]

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.2
        # for item in self.embeddings:
        self.embeddings_d.weight.data.uniform_(-initrange, initrange)
        # self.embeddings_d.weight.data.xavier_uniform_(tensor, gain=1.0)
        self.inter.data.uniform_(-initrange, initrange)




