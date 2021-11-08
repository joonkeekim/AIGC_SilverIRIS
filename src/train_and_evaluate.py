from torch.nn.modules import transformer
from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
from src.model_for_transformer import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time
import pdb
MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang,
                              num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang,
                              num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size, device): # edit
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.to(device)
        masked_index = masked_index.to(device)
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    # S x B x H -> (B x S) x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0), masked_index


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def stack_to_string(stack):
    op = ""
    for i in stack:
        if op == "":
            op = op + i
        else:
            op = op + ' ' + i
    return op


def sentence_from_indexes(lang, indexes):
    sent = []
    for ind in indexes:
        sent.append(lang.index2word[ind])
    return sent


def index_batch_to_words(input_batch, input_length, lang):
    '''
            Args:
                    input_batch: List of BS x Max_len
                    input_length: List of BS
            Return:
                    contextual_input: List of BS
    '''
    contextual_input = []
    # print(input_length)
    for i in range(len(input_batch)):
        # pdb.set_trace()
        contextual_input.append(stack_to_string(
            sentence_from_indexes(lang, input_batch[i][:input_length[i]])))

    return contextual_input


def train_tree_double(encoder_outputs, problem_output, all_nums_encoder_outputs, target, target_length,
                      output_lang, batch_size, padding_hidden, seq_mask,
                      num_mask, num_pos, num_order_pad, nums_stack_batch, unk,
                      encoder, numencoder, predict, generate, merge): # edit encoder_outputs 신경쓰기
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

#    copy_num_len = [len(_) for _ in num_pos]
#    num_size = max(copy_num_len)
#    nums_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
#                                                                        encoder.hidden_size)
#    all_nums_encoder_outputs = numencoder(nums_encoder_outputs, num_order_pad)
#    all_nums_encoder_outputs = all_nums_encoder_outputs.masked_fill_(masked_index.bool(), 0.0)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(
            target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.to(encoder_outputs.device)
        left_child, right_child, node_label = generate(
            current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx,
                                                      i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(
                        op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.to(encoder_outputs.device)
        target = target.to(encoder_outputs.device)

    # op_target = target < num_start
    loss = masked_cross_entropy(all_node_outputs, target, target_length, encoder_outputs.device)
    # loss = loss_0 + loss_1
    return loss  # , loss_0.item(), loss_1.item()


def train_attn_double(encoder_outputs, decoder_hidden, target, target_length,
                      output_lang, batch_size, seq_mask,
                      num_start, nums_stack_batch, unk,
                      decoder, beam_size, use_teacher_forcing): # edit encoder_outputs
    # Prepare input and output variables
    decoder_input = torch.LongTensor(
        [output_lang.word2index["SOS"]] * batch_size)

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(
        max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.to(encoder_outputs.device)

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        # pdb.set_trace()
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.to(encoder_outputs.device)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.to(encoder_outputs.device)
        beam_list.append(
            Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(
                batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(
                0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(
                max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.to(encoder_outputs.device)
                all_hidden = all_hidden.to(encoder_outputs.device)
                all_outputs = all_outputs.to(encoder_outputs.device)

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

#                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
#                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    decoder_input = decoder_input.to(encoder_outputs.device)

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

#                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                score = f.log_softmax(decoder_output, dim=1)
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx *
                            decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx *
                           batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk // decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.to(encoder_outputs.device)
                indices += temp_beam_pos.long() * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(
                    Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.to(encoder_outputs.device)
    # pdb.set_trace()
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length,
        encoder_outputs.device
    )

    return loss


def train_transformer_double(encoder_outputs, target, target_length,
                             output_lang, batch_size, input1_var,
                             num_start, nums_stack_batch, unk,
                             transformer_decoder, beam_size, use_teacher_forcing):
    ####### ---------------------------###
    # pdb.set_trace()
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size).unsqueeze(0).to(target.device)
    target_input = torch.cat((decoder_input, target), dim=0)
    # target_input : <s> N1 + N2 </s> <pad> <pad 0>
    # target : N1 + N2 </s>
    if random.random() < use_teacher_forcing:
        output = transformer_decoder.forward(
        encoder_outputs, target_input[:-1, :], input1_var, 0, output_lang.word2index['PAD'])
        loss = masked_cross_entropy(
            output.transpose(0, 1).contiguous(),  # -> batch x seq
            target.transpose(0, 1).contiguous(),  # -> batch x seq
            target_length,
            encoder_outputs.device
        )
    else:
        output = transformer_decoder.forward_topk( # target_length는 max로 고쳐주자.
            encoder_outputs, max(target_length), input1_var, 0, output_lang.word2index['PAD'])
        output_dim = output.shape[-1]
        loss = transformer_decoder.criterion(
            output.contiguous().view(-1, output_dim), target.contiguous().view(-1))
    
    # loss = transformer_decoder.criterion(
    #     output.contiguous().view(-1, output_dim), target.contiguous().view(-1))
    return loss


# def train_double(input1_batch, input2_batch, input_length, target1_batch, target1_length, target2_batch, target2_length,
#                  num_stack_batch, num_size_batch, generate_num1_ids, generate_num2_ids, copy_nums,
#                  encoder, numencoder, predict, generate, merge, decoder,
#                  encoder_optimizer, numencoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, decoder_optimizer,
#                  input_lang, output1_lang, output2_lang, num_pos_batch, num_order_batch, parse_graph_batch,
#                  beam_size=5, use_teacher_forcing=0.83, english=False):
# embedding, embedding_optimizer,
# add
def train_double(input1_text, input1_batch, input2_batch, input_length, target1_batch, target1_length, target2_batch, target2_length,
                 num_stack_batch, num_size_batch, generate_num1_ids, generate_num2_ids, copy_nums,
                 embedding, encoder, numencoder, predict, generate, merge, decoder, 
                 embedding_optimizer, encoder_optimizer, numencoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, decoder_optimizer, 
                 input_lang, output1_lang, output2_lang, num_pos_batch, num_order_batch, parse_graph_batch,
                 beam_size=5, use_teacher_forcing=0.83, english=False, device='cuda:0'):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    # pdb.set_trace()
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_num1_ids)
    for i in num_size_batch:
        d = i + len(generate_num1_ids)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    num_pos_pad = []
    max_num_pos_size = max(num_size_batch)
    for i in range(len(num_pos_batch)):
        temp = num_pos_batch[i] + [-1] * \
            (max_num_pos_size-len(num_pos_batch[i]))
        num_pos_pad.append(temp)
    num_pos_pad = torch.LongTensor(num_pos_pad)

    num_order_pad = []
    max_num_order_size = max(num_size_batch)
    for i in range(len(num_order_batch)):
        temp = num_order_batch[i] + [0] * \
            (max_num_order_size-len(num_order_batch[i]))
        num_order_pad.append(temp)
    num_order_pad = torch.LongTensor(num_order_pad)

    num_stack1_batch = copy.deepcopy(num_stack_batch)
    num_stack2_batch = copy.deepcopy(num_stack_batch)
    num_start2 = output2_lang.n_words - copy_nums - 2
    unk1 = output1_lang.word2index["UNK"]
    unk2 = output2_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input1_var = torch.LongTensor(input1_batch).transpose(0, 1)
    input2_var = torch.LongTensor(input2_batch).transpose(0, 1)
    target1 = torch.LongTensor(target1_batch).transpose(0, 1)
    target2 = torch.LongTensor(target2_batch).transpose(0, 1)
    parse_graph_pad = torch.LongTensor(parse_graph_batch)

    padding_hidden = torch.FloatTensor(
        [0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    # add
    embedding.train()
    encoder.train()
    numencoder.train()
    predict.train()
    generate.train()
    merge.train()
    decoder.train()
    # transformer_decoder.train()

    if USE_CUDA:
        input1_var = input1_var.to(device)
        input2_var = input2_var.to(device)
        seq_mask = seq_mask.to(device)
        padding_hidden = padding_hidden.to(device)
        num_mask = num_mask.to(device)
        num_pos_pad = num_pos_pad.to(device)
        num_order_pad = num_order_pad.to(device)
        parse_graph_pad = parse_graph_pad.to(device)

    # Zero gradients of both optimizers
    # add
    embedding_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    numencoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # transformer_decoder_optimizer.zero_grad()
    # Run words through encoder

    # add
    # pdb.set_trace()
    # contextual_input = index_batch_to_words(input1_batch, input_length, input_lang)
    input_seq1, token_ids = embedding(input1_text, max_len)
    input_seq1 = input_seq1.transpose(0, 1)
    # pdb.set_trace()
    encoder_outputs, encoder_hidden = encoder(
        input_seq1, input2_var, input_length, parse_graph_pad)
    copy_num_len = [len(_) for _ in num_pos_batch]
    num_size = max(copy_num_len)
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos_batch,
                                                                       batch_size, num_size, encoder.hidden_size,device)
    encoder_outputs, num_outputs, problem_output = numencoder(encoder_outputs, num_encoder_outputs,
                                                              num_pos_pad, num_order_pad)
    num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)

    # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    loss_0 = train_tree_double(encoder_outputs, problem_output, num_outputs, target1, target1_length,
                               output1_lang, batch_size, padding_hidden, seq_mask,
                               num_mask, num_pos_batch, num_order_pad, num_stack1_batch, unk1,
                               encoder, numencoder, predict, generate, merge)

    loss_1 = train_attn_double(encoder_outputs, decoder_hidden, target2, target2_length,
                               output2_lang, batch_size, seq_mask,
                               num_start2, num_stack2_batch, unk2,
                               decoder, beam_size, use_teacher_forcing)
    # loss_2 = train_transformer_double(input_seq1, target2.cuda(), target2_length,
    #                                   output2_lang, batch_size, input1_var,
    #                                   num_start2, num_stack2_batch, unk2,
    #                                   transformer_decoder, beam_size, use_teacher_forcing)
    loss = loss_0 + loss_1
    # loss = loss_2
    # pdb.set_trace()
    # print(loss.item())
    loss.backward()

    embedding_optimizer.step()
    encoder_optimizer.step()
    numencoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    decoder_optimizer.step()

    # transformer_decoder_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_tree_double(encoder_outputs, problem_output, all_nums_encoder_outputs,
                         output_lang, batch_size, padding_hidden, seq_mask, num_mask,
                         max_length, num_pos, num_order_pad,
                         encoder, numencoder, predict, generate, merge, beam_size):
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(
                torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.to(encoder_outputs.device)
                    left_child, right_child, node_label = generate(
                        current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(
                        TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(
                        TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0,
                                                          out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(
                            op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(
                        TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(
                        current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0]


def evaluate_attn_double(encoder_outputs, decoder_hidden,
                         output_lang, batch_size, seq_mask, max_length,
                         decoder, beam_size):
    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0]
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.to(encoder_outputs.device)
            all_hidden = all_hidden.to(encoder_outputs.device)
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                decoder_input = decoder_input.to(encoder_outputs.device)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx *
                        decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(
                Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0]

def evaluate_transformer_double(encoder_outputs, output_lang, batch_size,
                                    max_length, input1_var,
                                    transformer_decoder):

    pass

# def evaluate_double(input1_batch, input2_batch, input_length, generate_num1_ids, generate_num2_ids,
#                     encoder, numencoder, predict, generate, merge, decoder,
#                     input_lang, output1_lang, output2_lang, num_pos_batch, num_order_batch, parse_graph_batch,
#                     beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
def evaluate_double(input_text1, input1_batch, input2_batch, input_length, generate_num1_ids, generate_num2_ids,
                    embedding, encoder, numencoder, predict, generate, merge, decoder, 
                    input_lang, output1_lang, output2_lang, num_pos_batch, num_order_batch, parse_graph_batch,
                    beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH,device='cuda:0'):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_pos_pad = torch.LongTensor([num_pos_batch])
    num_order_pad = torch.LongTensor([num_order_batch])
    parse_graph_pad = torch.LongTensor(parse_graph_batch)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input1_var = torch.LongTensor(input1_batch).unsqueeze(1)
    input2_var = torch.LongTensor(input2_batch).unsqueeze(1)
    max_len = input_length
    num_mask = torch.ByteTensor(
        1, len(num_pos_batch) + len(generate_num1_ids)).fill_(0)

    # Set to not-training mode to disable dropout
    # add
    embedding.eval()
    encoder.eval()
    numencoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    decoder.eval()

    padding_hidden = torch.FloatTensor(
        [0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input1_var = input1_var.to(device)
        input2_var = input2_var.to(device)
        seq_mask = seq_mask.to(device)
        padding_hidden = padding_hidden.to(device)
        num_mask = num_mask.to(device)
        num_pos_pad = num_pos_pad.to(device)
        num_order_pad = num_order_pad.to(device)
        parse_graph_pad = parse_graph_pad.to(device)
    # Run words through encoder
    # pdb.set_trace()
    input_seq1, input_len1 = embedding([input_text1], max_len)
    input_seq1 = input_seq1.transpose(0, 1)
    encoder_outputs, encoder_hidden = encoder(
        input_seq1, input2_var, [input_length], parse_graph_pad)
    num_size = len(num_pos_batch)
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, [num_pos_batch], batch_size,
                                                                       num_size, encoder.hidden_size, device)
    encoder_outputs, num_outputs, problem_output = numencoder(encoder_outputs, num_encoder_outputs,
                                                              num_pos_pad, num_order_pad)
    # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    tree_beam = evaluate_tree_double(encoder_outputs, problem_output, num_outputs,
                                     output1_lang, batch_size, padding_hidden, seq_mask, num_mask,
                                     max_length, num_pos_batch, num_order_pad,
                                     encoder, numencoder, predict, generate, merge, beam_size)

    attn_beam = evaluate_attn_double(encoder_outputs, decoder_hidden,
                                     output2_lang, batch_size, seq_mask, max_length,
                                     decoder, beam_size)
                                     
    # trans_topk = evaluate_transformer_double(input_seq1, output2_lang, batch_size, max_length, input1_var,
    #                                  transformer_decoder)
    if tree_beam.score >= attn_beam.score:
        return "tree", tree_beam.out, tree_beam.score
    else:
        return "attn", attn_beam.all_output, attn_beam.score

def evaluate_gru(input_text1, input1_batch, input2_batch, input_length, generate_num1_ids, generate_num2_ids,
                    embedding, encoder, numencoder, predict, generate, merge, decoder, 
                    input_lang, output1_lang, output2_lang, num_pos_batch, num_order_batch, parse_graph_batch,
                    beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH,device='cuda:0'):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_pos_pad = torch.LongTensor([num_pos_batch])
    num_order_pad = torch.LongTensor([num_order_batch])
    parse_graph_pad = torch.LongTensor(parse_graph_batch)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input1_var = torch.LongTensor(input1_batch).unsqueeze(1)
    input2_var = torch.LongTensor(input2_batch).unsqueeze(1)
    max_len = input_length
    num_mask = torch.ByteTensor(
        1, len(num_pos_batch) + len(generate_num1_ids)).fill_(0)

    # Set to not-training mode to disable dropout
    # add
    embedding.eval()
    encoder.eval()
    numencoder.eval()
    # predict.eval()
    # generate.eval()
    # merge.eval()
    decoder.eval()

    # padding_hidden = torch.FloatTensor(
    #     [0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input1_var = input1_var.to(device)
        input2_var = input2_var.to(device)
        seq_mask = seq_mask.to(device)
        # padding_hidden = padding_hidden.to(device)
        num_mask = num_mask.to(device)
        num_pos_pad = num_pos_pad.to(device)
        num_order_pad = num_order_pad.to(device)
        parse_graph_pad = parse_graph_pad.to(device)
    # Run words through encoder
    # pdb.set_trace()
    input_seq1, input_len1 = embedding([input_text1], max_len)
    input_seq1 = input_seq1.transpose(0, 1)
    encoder_outputs, encoder_hidden = encoder(
        input_seq1, input2_var, [input_length], parse_graph_pad)
    num_size = len(num_pos_batch)
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, [num_pos_batch], batch_size,
                                                                       num_size, encoder.hidden_size, device)
    encoder_outputs, num_outputs, problem_output = numencoder(encoder_outputs, num_encoder_outputs,
                                                              num_pos_pad, num_order_pad)
    # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # tree_beam = evaluate_tree_double(encoder_outputs, problem_output, num_outputs,
    #                                  output1_lang, batch_size, padding_hidden, seq_mask, num_mask,
    #                                  max_length, num_pos_batch, num_order_pad,
    #                                  encoder, numencoder, predict, generate, merge, beam_size)

    attn_beam = evaluate_attn_double(encoder_outputs, decoder_hidden,
                                     output2_lang, batch_size, seq_mask, max_length,
                                     decoder, beam_size)
                                     
    # trans_topk = evaluate_transformer_double(input_seq1, output2_lang, batch_size, max_length, input1_var,
    #                                  transformer_decoder)
    # if tree_beam.score >= attn_beam.score:
    #     return "tree", tree_beam.out, tree_beam.score
    # else:
    #     return "attn", attn_beam.all_output, attn_beam.score
    return "attn", attn_beam.all_output, attn_beam.score

def evaluate_tree(input_text1, input1_batch, input2_batch, input_length, generate_num1_ids, generate_num2_ids,
                    embedding, encoder, numencoder, predict, generate, merge, decoder, 
                    input_lang, output1_lang, output2_lang, num_pos_batch, num_order_batch, parse_graph_batch,
                    beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH,device='cuda:0'):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_pos_pad = torch.LongTensor([num_pos_batch])
    num_order_pad = torch.LongTensor([num_order_batch])
    parse_graph_pad = torch.LongTensor(parse_graph_batch)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input1_var = torch.LongTensor(input1_batch).unsqueeze(1)
    input2_var = torch.LongTensor(input2_batch).unsqueeze(1)
    max_len = input_length
    num_mask = torch.ByteTensor(
        1, len(num_pos_batch) + len(generate_num1_ids)).fill_(0)

    # add
    embedding.eval()
    encoder.eval()
    numencoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor(
        [0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input1_var = input1_var.to(device)
        input2_var = input2_var.to(device)
        seq_mask = seq_mask.to(device)
        padding_hidden = padding_hidden.to(device)
        num_mask = num_mask.to(device)
        num_pos_pad = num_pos_pad.to(device)
        num_order_pad = num_order_pad.to(device)
        parse_graph_pad = parse_graph_pad.to(device)
    # Run words through encoder
    # pdb.set_trace()
    input_seq1, input_len1 = embedding([input_text1], max_len)
    input_seq1 = input_seq1.transpose(0, 1)
    encoder_outputs, encoder_hidden = encoder(
        input_seq1, input2_var, [input_length], parse_graph_pad)
    num_size = len(num_pos_batch)
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, [num_pos_batch], batch_size,
                                                                       num_size, encoder.hidden_size, device)
    encoder_outputs, num_outputs, problem_output = numencoder(encoder_outputs, num_encoder_outputs,
                                                              num_pos_pad, num_order_pad)

    tree_beam = evaluate_tree_double(encoder_outputs, problem_output, num_outputs,
                                     output1_lang, batch_size, padding_hidden, seq_mask, num_mask,
                                     max_length, num_pos_batch, num_order_pad,
                                     encoder, numencoder, predict, generate, merge, beam_size)

    return "tree", tree_beam.out, tree_beam.score
