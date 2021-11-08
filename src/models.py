import copy
import torch
import torch.nn as nn
import pdb


def replace_masked_values(tensor, mask, replace_with):
    return tensor.masked_fill((1 - mask).bool(), replace_with)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:,
                                                             :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        # torch.cat(dd)
        energy_in = torch.cat((hidden, encoder_outputs),
                              2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(
            self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(
            max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size,
                          hidden_size, n_layers, dropout=dropout)
        # self.gru2 = nn.Transformer(d_model = hidden_size + embedding_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):

        # Get the embedding of the current input word (last output word)
        # pdb.set_trace()
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(
            1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(
            last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1))  # B x S=1 x N

        rnn_output, hidden = self.gru(
            torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(
            torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings),
                              2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs),
                              2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(
            max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)

# edit


class EncoderSeq(nn.Module):
    def __init__(self, input1_size, input2_size,  embedding1_size,
                 embedding2_size, hidden_size, n_layers=2, hop_size=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.embedding1_size = embedding1_size
        self.embedding2_size = embedding2_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.hop_size = hop_size

        # self.embedding1 = embed_model
        self.embedding2 = nn.Embedding(
            input2_size, embedding2_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding1_size+embedding2_size, hidden_size,
                          n_layers, dropout=dropout, bidirectional=True)
        self.parse_gnn = clones(Parse_Graph_Module(hidden_size), hop_size)

    def forward(self, embedded1, input2_var, input_length, parse_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # embedded1 = self.embedding1(input1_var)  # S x B x E
        # pdb.set_trace()
        embedded2 = self.embedding2(input2_var)
        embedded = torch.cat((embedded1, embedded2), dim=2)
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_length)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        pade_outputs = pade_outputs[:, :, :self.hidden_size] + \
            pade_outputs[:, :, self.hidden_size:]  # S x B x H
        pade_outputs = pade_outputs.transpose(0, 1)
        for i in range(self.hop_size):
            pade_outputs = self.parse_gnn[i](pade_outputs, parse_graph[:, 2])
        pade_outputs = pade_outputs.transpose(0, 1)
#        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]

        return pade_outputs, pade_hidden


class Parse_Graph_Module(nn.Module):
    def __init__(self, hidden_size):
        super(Parse_Graph_Module, self).__init__()

        self.hidden_size = hidden_size
        self.node_fc1 = nn.Linear(hidden_size, hidden_size)
        self.node_fc2 = nn.Linear(hidden_size, hidden_size)
        self.node_out = nn.Linear(hidden_size * 2, hidden_size)

    def normalize(self, graph, symmetric=True):
        d = graph.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(graph).mm(D)
        else:
            D = torch.diag(torch.pow(d, -1))
            return D.mm(graph)

    def forward(self, node, graph):
        graph = graph.float()
        batch_size = node.size(0)
        for i in range(batch_size):
            graph[i] = self.normalize(graph[i])

        node_info = torch.relu(self.node_fc1(torch.matmul(graph, node)))
        node_info = torch.relu(self.node_fc2(torch.matmul(graph, node_info)))

        agg_node_info = torch.cat((node, node_info), dim=2)
        agg_node_info = torch.relu(self.node_out(agg_node_info))

        return agg_node_info


class NumEncoder(nn.Module):
    def __init__(self, node_dim, hop_size=2):
        super(NumEncoder, self).__init__()

        self.node_dim = node_dim
        self.hop_size = hop_size
        self.num_gnn = clones(Num_Graph_Module(node_dim), hop_size)

    def forward(self, encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad):
        num_embedding = num_encoder_outputs.clone()
        batch_size = num_embedding.size(0)
        num_mask = (num_pos_pad > -1).long()
        node_mask = (num_order_pad > 0).long()
        greater_graph_mask = num_order_pad.unsqueeze(-1).expand(batch_size, -1, num_order_pad.size(-1)) > \
            num_order_pad.unsqueeze(1).expand(
                batch_size, num_order_pad.size(-1), -1)
        lower_graph_mask = num_order_pad.unsqueeze(-1).expand(batch_size, -1, num_order_pad.size(-1)) <= \
            num_order_pad.unsqueeze(1).expand(
                batch_size, num_order_pad.size(-1), -1)
        greater_graph_mask = greater_graph_mask.long()
        lower_graph_mask = lower_graph_mask.long()

        diagmat = torch.diagflat(torch.ones(num_embedding.size(
            1), dtype=torch.long, device=num_embedding.device))
        diagmat = diagmat.unsqueeze(0).expand(num_embedding.size(0), -1, -1)
        graph_ = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1-diagmat)
        graph_greater = graph_ * greater_graph_mask + diagmat
        graph_lower = graph_ * lower_graph_mask + diagmat

        for i in range(self.hop_size):
            num_embedding = self.num_gnn[i](
                num_embedding, graph_greater, graph_lower)
#        gnn_info_vec = torch.zeros((batch_size, 1, encoder_outputs.size(-1)),
#                                   dtype=torch.float, device=num_embedding.device)
#        gnn_info_vec = torch.cat((encoder_outputs.transpose(0, 1), gnn_info_vec), dim=1)
        gnn_info_vec = torch.zeros((batch_size, encoder_outputs.size(0)+1, encoder_outputs.size(-1)),
                                   dtype=torch.float, device=num_embedding.device)

        clamped_number_indices = replace_masked_values(
            num_pos_pad, num_mask, gnn_info_vec.size(1)-1)
        gnn_info_vec.scatter_(1, clamped_number_indices.unsqueeze(
            -1).expand(-1, -1, num_embedding.size(-1)), num_embedding)
        gnn_info_vec = gnn_info_vec[:, :-1, :]
        gnn_info_vec = gnn_info_vec.transpose(0, 1)
        gnn_info_vec = encoder_outputs + gnn_info_vec
        num_embedding = num_encoder_outputs + num_embedding
        problem_output = torch.max(gnn_info_vec, 0).values

        return gnn_info_vec, num_embedding, problem_output


class Num_Graph_Module(nn.Module):
    def __init__(self, node_dim):
        super(Num_Graph_Module, self).__init__()

        self.node_dim = node_dim
        self.node1_fc1 = nn.Linear(node_dim, node_dim)
        self.node1_fc2 = nn.Linear(node_dim, node_dim)
        self.node2_fc1 = nn.Linear(node_dim, node_dim)
        self.node2_fc2 = nn.Linear(node_dim, node_dim)
        self.graph_weight = nn.Linear(node_dim * 4, node_dim)
        self.node_out = nn.Linear(node_dim * 2, node_dim)

    def normalize(self, graph, symmetric=True):
        d = graph.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(graph).mm(D)
        else:
            D = torch.diag(torch.pow(d, -1))
            return D.mm(graph)

    def forward(self, node, graph1, graph2):
        graph1 = graph1.float()
        graph2 = graph2.float()
        batch_size = node.size(0)

        for i in range(batch_size):
            graph1[i] = self.normalize(graph1[i], False)
            graph2[i] = self.normalize(graph2[i], False)

        node_info1 = torch.relu(self.node1_fc1(torch.matmul(graph1, node)))
        node_info1 = torch.relu(self.node1_fc2(
            torch.matmul(graph1, node_info1)))
        node_info2 = torch.relu(self.node2_fc1(torch.matmul(graph2, node)))
        node_info2 = torch.relu(self.node2_fc2(
            torch.matmul(graph2, node_info2)))
        gate = torch.cat((node_info1, node_info2, node_info1 +
                          node_info2, node_info1-node_info2), dim=2)
        gate = torch.sigmoid(self.graph_weight(gate))
        node_info = gate * node_info1 + (1-gate) * node_info2
        agg_node_info = torch.cat((node, node_info), dim=2)
        agg_node_info = torch.relu(self.node_out(agg_node_info))

        return agg_node_info


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(
            torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(
            0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(
            encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(
            *repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat(
            (embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1),
                               embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(
            torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(
            torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(
            torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(
            torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(
            torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(
            torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree
