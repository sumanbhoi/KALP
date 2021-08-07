import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution




'''
Transformer based Lab test response prediction'''

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBased(nn.Module):
    def __init__(self, ntoken, ninp=64, nhead=2, nhid=64, nlayers=6, dropout=0.5): # , device=torch.device('cpu:0')):
        super(TransformerBased, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        # self.device = device
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

    # def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
    #     super(TransformerBased, self).__init__()
    #     # K = len(vocab_size)
    #     K = 2
    #     ''' Length here is 3 (3 types of codes)'''
    #     self.K = K
    #     self.vocab_size = vocab_size
    #     self.device = device
    #     # self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
    #     # self.ddi_in_memory = ddi_in_memory
    #     # self.embeddings = nn.ModuleList(
    #     #     [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
    #     self.embeddings = nn.ModuleList(
    #         [nn.Embedding(vocab_size[1], emb_dim)])
    #     # print("structure", self.embeddings)
    #     self.dropout = nn.Dropout(p=0.4)
    #
    #     self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])
    #
    #     # self.query = nn.Sequential(
    #     #     nn.ReLU(),
    #     #     nn.Linear(emb_dim * 4, emb_dim),
    #     # )
    #
    #     self.query = nn.Sequential(
    #         nn.ReLU(),
    #         nn.Linear(emb_dim * 2, emb_dim),
    #     )
    #
    #     # self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
    #     # self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
    #     # self.inter = nn.Parameter(torch.FloatTensor(1))
    #
    #     # self.output = nn.Sequential(
    #     #     nn.ReLU(),
    #     #     nn.Linear(emb_dim * 3, emb_dim * 2),
    #     #     nn.ReLU(),
    #     #     nn.Linear(emb_dim * 2, vocab_size[2])
    #     # )
    #     self.output = nn.Linear(emb_dim, 1)         # For the changed version of gamenet
    #     self.init_weights()
    #
    # def forward(self, input):
    #     # input (adm, 3, codes)
    #
    #     # generate medical embeddings and queries
    #     i1_seq = []
    #     i2_seq = []
    #     def mean_embedding(embedding):
    #         return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
    #     for adm in input:
    #         i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
    #         """ [1,1,64]"""
    #         # i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
    #         i1_seq.append(i1)
    #         # i2_seq.append(i2)
    #         # a = (self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))
    #     # print("Embedding", a)
    #     # print("Embedding Size a", np.shape(a))
    #     i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
    #     # i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)
    #
    #     o1, h1 = self.encoders[0](
    #         i1_seq
    #     ) # o1:(1, seq, dim*2) hi:(1,1,dim*2)
    #     # o2, h2 = self.encoders[1](
    #     #     i2_seq
    #     # )
    #     # patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
    #     patient_representations = o1.squeeze(dim=0)
    #     # print('shape', np.shape(patient_representations))
    #     # print(np.shape(patient_representations1))
    #     queries = self.query(patient_representations) # (seq, dim)
    #
    #     # graph memory module
    #     '''I:generate current input'''
    #     query = queries[-1:] # (1,dim)
    #
    #     # '''G:generate graph memory bank and insert history information'''
    #     # if self.ddi_in_memory:
    #     #     drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
    #     #     # print('drug memory', drug_memory)
    #     #     # print('size of memory', np.shape(drug_memory))
    #     # else:
    #     #     drug_memory = self.ehr_gcn()
    #     # # print("ehr", (self.ehr_gcn()))
    #     # # print("ddi", (self.ddi_gcn()))
    #     # # print("drug memory", (drug_memory))
    #
    #     # if len(input) > 1:
    #     #     history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)
    #     #
    #     #     history_values = np.zeros((len(input)-1, self.vocab_size[2]))
    #     #     for idx, adm in enumerate(input):
    #     #         if idx == len(input)-1:
    #     #             break
    #     #         history_values[idx, adm[2]] = 1
    #     #     history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)
    #     #     # print('history size', (history_keys))
    #
    #     # '''O:read from global memory bank and dynamic memory bank'''
    #     # key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
    #     # fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)
    #
    #     # if len(input) > 1:
    #     #     visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
    #     #     weighted_values = visit_weight.mm(history_values) # (1, size)
    #     #     fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
    #     # else:
    #     #     fact2 = fact1
    #     '''R:convert O and predict'''
    #     # output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)
    #     output = self.output(query)
    #     return output
    #     # if self.training:
    #     #     neg_pred_prob = F.linear(output)
    #     #     # print('neg_pred_prob', np.shape(neg_pred_prob))
    #     #     neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
    #     #     batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
    #     #     # print('ddi_adj',np.shape(self.tensor_ddi_adj))
    #     #     return output, batch_neg
    #     # else:
    #     #     return output
    #
    # def init_weights(self):
    #     """Initialize weights."""
    #     initrange = 0.1
    #     for item in self.embeddings:
    #         item.weight.data.uniform_(-initrange, initrange)
    #
    #     # self.inter.data.uniform_(-initrange, initrange)






'''
Our model
'''
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GAMENet(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(GAMENet, self).__init__()
        # K = len(vocab_size)
        K = 2
        ''' Length here is 3 (3 types of codes)'''
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        # self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        # self.ddi_in_memory = ddi_in_memory
        # self.embeddings = nn.ModuleList(
        #     [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[1], emb_dim)])
        # print("structure", self.embeddings)
        self.dropout = nn.Dropout(p=0.4)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])

        # self.query = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(emb_dim * 4, emb_dim),
        # )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        # self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        # self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        # self.inter = nn.Parameter(torch.FloatTensor(1))

        # self.output = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(emb_dim * 3, emb_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim * 2, vocab_size[2])
        # )
        self.output = nn.Linear(emb_dim, 1)         # For the changed version of gamenet
        self.init_weights()

    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            """ [1,1,64]"""
            # i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            # i2_seq.append(i2)
            # a = (self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))
        # print("Embedding", a)
        # print("Embedding Size a", np.shape(a))
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        # i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        ) # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        # o2, h2 = self.encoders[1](
        #     i2_seq
        # )
        # patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        patient_representations = o1.squeeze(dim=0)
        # print('shape', np.shape(patient_representations))
        # print(np.shape(patient_representations1))
        queries = self.query(patient_representations) # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:] # (1,dim)

        # '''G:generate graph memory bank and insert history information'''
        # if self.ddi_in_memory:
        #     drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        #     # print('drug memory', drug_memory)
        #     # print('size of memory', np.shape(drug_memory))
        # else:
        #     drug_memory = self.ehr_gcn()
        # # print("ehr", (self.ehr_gcn()))
        # # print("ddi", (self.ddi_gcn()))
        # # print("drug memory", (drug_memory))

        # if len(input) > 1:
        #     history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)
        #
        #     history_values = np.zeros((len(input)-1, self.vocab_size[2]))
        #     for idx, adm in enumerate(input):
        #         if idx == len(input)-1:
        #             break
        #         history_values[idx, adm[2]] = 1
        #     history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)
        #     # print('history size', (history_keys))

        # '''O:read from global memory bank and dynamic memory bank'''
        # key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        # fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        # if len(input) > 1:
        #     visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
        #     weighted_values = visit_weight.mm(history_values) # (1, size)
        #     fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        # else:
        #     fact2 = fact1
        '''R:convert O and predict'''
        # output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)
        output = self.output(query)
        return output
        # if self.training:
        #     neg_pred_prob = F.linear(output)
        #     # print('neg_pred_prob', np.shape(neg_pred_prob))
        #     neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        #     batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
        #     # print('ddi_adj',np.shape(self.tensor_ddi_adj))
        #     return output, batch_neg
        # else:
        #     return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        # self.inter.data.uniform_(-initrange, initrange)

'''
DMNC
'''
class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.3)

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
            independent_linears=False
        ) for _ in range(K - 1)])

        self.decoder = nn.GRU(emb_dim + emb_dim * 2, emb_dim * 2,
                              batch_first=True)  # input: (y, r1, r2,) hidden: (hidden1, hidden2)
        self.interface_weighting = nn.Linear(emb_dim * 2, 2 * (emb_dim + 1 + 3))  # 2 read head (key, str, mode)
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)

    def forward(self, input, i1_state=None, i2_state=None, h_n=None, max_len=20):
        # input (3, code)
        i1_input_tensor = self.embeddings[0](
            torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
        i2_input_tensor = self.embeddings[1](
            torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        o1, (ch1, m1, r1) = \
            self.encoders[0](i1_input_tensor, (None, None, None) if i1_state is None else i1_state)
        o2, (ch2, m2, r2) = \
            self.encoders[1](i2_input_tensor, (None, None, None) if i2_state is None else i2_state)

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + input[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)

        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(read_key, read_str, read_mode, m_hidden)
        return read_vectors, hidden

    def decode_read_variable(self, input):
        w = 64
        r = 2
        b = input.size(0)

        input = self.interface_weighting(input)
        # r read keys (b * w * r)
        read_keys = F.tanh(input[:, :r * w].contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(input[:, r * w:r * w + r].contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(input[:, (r * w + r):].contiguous().view(b, r, 3), -1)
        return read_keys, read_strengths, read_modes


'''
Leap
'''
class Leap(nn.Module):
    def __init__(self, voc_size, emb_dim=128, device=torch.device('cpu:0')):
        super(Leap, self).__init__()
        self.voc_size = voc_size
        self.device = device
        self.SOS_TOKEN = voc_size[2]
        self.END_TOKEN = voc_size[2]+1

        self.enc_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim, ),
            nn.Dropout(0.3)
        )
        self.dec_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 2, emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim*2, 1)

        self.output = nn.Linear(emb_dim, voc_size[2]+2)


    def forward(self, input, max_len=20):
        device = self.device
        # input (3, codes)
        input_tensor = torch.LongTensor(input[0]).to(device)
        # (len, dim)
        input_embedding = self.enc_embedding(input_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        output_logits = []
        hidden_state = None
        if self.training:
            for med_code in [self.SOS_TOKEN] + input[2]:
                dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)

                if hidden_state is None:
                    hidden_state = dec_input

                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1) # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1) # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1) # (1, len)
                # input_embedding = attn_weight.mm(input_embedding) # (1, dim)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)

                # _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))

                hidden_state = hidden_state.squeeze(dim=0) # (1,dim)

                output_logits.append(self.output(F.relu(hidden_state)))

            return torch.cat(output_logits, dim=0)

        else:
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)  # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)  # (1, len)
                input = attn_weight.mm(input_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(torch.cat([input, dec_input], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                # print('output', np.shape(output))
                topv, topi = output.data.topk(1)
                # print(topv, topi)
                output_logits.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(output_logits, dim=0)

'''
Retain
'''
class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.3)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            input_tmp.extend(list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1]))
            if len(input_tmp) < max_len:
                input_tmp.extend( [self.input_len]*(max_len - len(input_tmp)) )

            input_np.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_np).to(device)) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)



        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)

'''
RF in train_LR.py
'''


