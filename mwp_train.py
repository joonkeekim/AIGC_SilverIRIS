import os
import time
import torch.optim
import torch.nn as nn
print(torch.__version__)
# from src.model_for_transformer import *
from src.logger import *
from src.models import *
from src.train_and_evaluate import *
from src.expressions_transfer import *
from src.text_utils import *
from src.contextual_embeddings import *
from tqdm import tqdm

with open('data/mwp_data/ape_math_edit.json', encoding="utf-8") as f:
    data = json.load(f)

pairs, generate_nums, copy_nums = transfer_num(data)

import random
temp_pairs = []
f_cnt = 0
for p in tqdm(pairs):
    # pdb.set_trace()
    res = p[6]
    pos = [r[3] for r in res]
    arcs = [r[2] for r in res] # parse tree
    parse_tree = []
    cnt = 0
    parse_tree = [arc-1 if arc != -1 else arc for arc in arcs]
    # print(len(parse_tree),max(parse_tree))
    if len(p[1]) != len(parse_tree):
        f_cnt += 1
        continue
    temp_pairs.append((p[0], p[1], pos, parse_tree, 
                            p[2], p[3], p[4], p[5]))

print(f"{f_cnt} pairs are ignored by disorder")
pairs = temp_pairs

pairs = filter_data(pairs)
pairs = filter_with_input_length(pairs)

num_test_data = 3000
random.shuffle(pairs)
pairs_trained = pairs[num_test_data:]
pairs_tested = pairs[:num_test_data]

save_lang(pairs_trained,'pickles/pairs_trained.pickle')
save_lang(pairs_tested,'pickles/pairs_tested.pickle')
save_lang(generate_nums,'pickles/generate_nums.pickle')
save_lang(copy_nums,'pickles/copy_nums.pickle')
# pairs_trained = load_lang('pickles/pairs_trained.pickle')
# pairs_tested = load_lang('pickles/pairs_tested.pickle')
# generate_nums = load_lang('pickles/generate_nums.pickle')
# copy_nums = load_lang('pickles/copy_nums.pickle')

elogger = Logger("MultiMath_")

input1_lang, input2_lang, output1_lang, output2_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 3, generate_nums, copy_nums)

save_lang(input1_lang,'pickles/input1_lang.pickle')
save_lang(input2_lang,'pickles/input2_lang.pickle')
save_lang(output1_lang,'pickles/output1_lang.pickle')
save_lang(output2_lang,'pickles/output2_lang.pickle')
save_lang(train_pairs,'pickles/train_pairs.pickle')
save_lang(test_pairs,'pickles/test_pairs.pickle')

# input1_lang = load_lang('pickles/input1_lang.pickle')
# input2_lang = load_lang('pickles/input2_lang.pickle')
# output1_lang = load_lang('pickles/output1_lang.pickle')
# output2_lang = load_lang('pickles/output2_lang.pickle')
# train_pairs = load_lang('pickles/train_pairs.pickle')
# test_pairs = load_lang('pickles/test_pairs.pickle')
# generate_nums = load_lang('pickles/generate_nums.pickle')
# copy_nums = load_lang('pickles/copy_nums.pickle')

batch_size = 64
embedding_size = 1024
hidden_size = 512
n_epochs = 80
learning_rate = 3e-4
weight_decay = 1e-5
beam_size = 5
n_layers = 2
hop_size = 2
emb_lr = 3e-5
emb_name = 'pretrained_mlm_nsp'
device = 'cuda:1'
freeze_emb = False
FROM_FINETUING = True

# Initialize models
embedding = BertEncoder(emb_name, device, freeze_emb)
encoder = EncoderSeq(input1_size=input1_lang.n_words, input2_size=input2_lang.n_words, 
                        embedding1_size=embedding_size, embedding2_size=embedding_size//4, 
                        hidden_size=hidden_size, n_layers=n_layers, hop_size=hop_size)
numencoder = NumEncoder(node_dim=hidden_size, hop_size=hop_size)
predict = Prediction(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                        input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size,
                            input_size=output2_lang.n_words, output_size=output2_lang.n_words, n_layers=n_layers)
# transformer_decoder = TransformerModel(hidden_size=hidden_size, embedding_size=embedding_size, input_size=output2_lang.n_words, output_size=output2_lang.n_words,device=device)

if FROM_FINETUING:
    embedding.load_state_dict(torch.load('models/embedding.pt'))
    encoder.load_state_dict(torch.load('models/encoder.pt'))
    numencoder.load_state_dict(torch.load('models/numencoder.pt'))
    predict.load_state_dict(torch.load('models/predict.pt'))
    generate.load_state_dict(torch.load('models/generate.pt'))
    merge.load_state_dict(torch.load('models/merge.pt'))
    decoder.load_state_dict(torch.load('models/decoder.pt'))
    print('load successfully')



embedding_optimizer = torch.optim.AdamW(embedding.parameters(), lr=emb_lr, weight_decay=weight_decay)
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate,weight_decay=weight_decay)
numencoder_optimizer = torch.optim.AdamW(numencoder.parameters(), lr=learning_rate,weight_decay=weight_decay)
predict_optimizer = torch.optim.AdamW(predict.parameters(), lr=learning_rate,weight_decay=weight_decay)
generate_optimizer = torch.optim.AdamW(generate.parameters(), lr=learning_rate,weight_decay=weight_decay)
merge_optimizer = torch.optim.AdamW(merge.parameters(), lr=learning_rate,weight_decay=weight_decay)
decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate,weight_decay=weight_decay)
# transformer_decoder_optimizer = torch.optim.Adam(transformer_decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)


embedding_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(embedding_optimizer, T_0=8, T_mult=1, eta_min=1e-7)
encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(encoder_optimizer, T_0=8, T_mult=1, eta_min=1e-6)
numencoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(numencoder_optimizer, T_0=8, T_mult=1, eta_min=1e-6)
predict_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(predict_optimizer, T_0=8, T_mult=1, eta_min=1e-6)
generate_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(generate_optimizer, T_0=8, T_mult=1, eta_min=1e-6)
merge_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(merge_optimizer, T_0=8, T_mult=1, eta_min=1e-6)
decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(decoder_optimizer, T_0=8, T_mult=1, eta_min=1e-6)
# transformer_decoder_scheduler = torch.optim.lr_scheduler.StepLR(transformer_decoder_optimizer, step_size=20, gamma=0.5)

# Move models to GPU

if USE_CUDA:
    embedding.to(device)
    encoder.to(device)
    numencoder.to(device)
    predict.to(device)
    generate.to(device)
    merge.to(device)
    decoder.to(device)
    # transformer_decoder.to(device)

# elogger.log(str(encoder))
# elogger.log(str(numencoder))
# elogger.log(str(predict))
# elogger.log(str(generate))
# elogger.log(str(merge))
# elogger.log(str(decoder))

generate_num1_ids = []
generate_num2_ids = []
for num in generate_nums:
    generate_num1_ids.append(output1_lang.word2index[num])
    generate_num2_ids.append(output2_lang.word2index[num])

best_acc =0
for epoch in range(n_epochs):
    loss_total = 0
    id_batches, input1_texts, input1_batches, input2_batches, input_lengths, output1_batches, output1_lengths, output2_batches, output2_lengths, \
    nums_batches, num_stack_batches, num_pos_batches, num_order_batches, num_size_batches, parse_graph_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in tqdm(range(len(input_lengths))):
        loss = train_double(
            input1_texts[idx],input1_batches[idx], input2_batches[idx], input_lengths[idx], output1_batches[idx], output1_lengths[idx], output2_batches[idx], output2_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num1_ids, generate_num2_ids, copy_nums,
            embedding, encoder, numencoder, predict, generate, merge, decoder,
            embedding_optimizer, encoder_optimizer, numencoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, decoder_optimizer,
            input1_lang, output1_lang, output2_lang, num_pos_batches[idx], num_order_batches[idx], parse_graph_batches[idx], 
            beam_size=5, use_teacher_forcing=0.83, english=False,device=device)
        loss_total += loss

    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    # elogger.log("epoch: %d, loss: %.4f" % (epoch+1, loss_total/len(input_lengths)))


    value_ac = 0
    equation_ac = 0
    eval_total = 0
    result_list = []
    start = time.time()
     
    for test_batch in tqdm(test_pairs):
        parse_graph = get_parse_graph_batch([test_batch[5]], [test_batch[4]])
        result_type, test_res, score = evaluate_double(test_batch[1], test_batch[2], test_batch[3], test_batch[5], generate_num1_ids, generate_num2_ids,
                                                embedding, encoder, numencoder, predict, generate, merge, decoder,
                                                input1_lang, output1_lang, output2_lang, test_batch[11], test_batch[13], parse_graph, beam_size=beam_size,device=device)
        if result_type == "tree":
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[6], output1_lang, test_batch[10], test_batch[12])
            result = out_expression_list(test_res, output1_lang, test_batch[10])
            result_list.append([test_batch[0], "tree", result, score])
        else:
            if test_res[-1] == output2_lang.word2index["EOS"]:
                test_res = test_res[:-1]
            val_ac, equ_ac, _, _ = compute_postfix_tree_result(test_res, test_batch[8][:-1], output2_lang, test_batch[10], test_batch[12])
            result = out_expression_list(test_res, output2_lang, test_batch[10])
            result_list.append([test_batch[0], "attn", result, score])

        if val_ac:
            value_ac += 1
        if equ_ac:
            equation_ac += 1
        eval_total += 1
    if value_ac > best_acc:
        print('best acc renewed')
        best_acc = value_ac
        torch.save(embedding.state_dict(),'models_gpu1/embedding.pt')
        torch.save(encoder.state_dict(), "models_gpu1/encoder.pt")
        torch.save(numencoder.state_dict(), "models_gpu1/numencoder.pt")
        torch.save(predict.state_dict(), "models_gpu1/predict.pt")
        torch.save(generate.state_dict(), "models_gpu1/generate.pt")
        torch.save(merge.state_dict(), "models_gpu1/merge.pt")
        torch.save(decoder.state_dict(), "models_gpu1/decoder.pt")
    print(equation_ac, value_ac, eval_total)
    print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total, "best_acc", best_acc)
    print("testing time", time_since(time.time() - start))
    print("------------------------------------------------------")
    
    # write_data_json(result_list, "results/result_.json")
    # elogger.log("epoch: %d, test_equ_acc: %.4f, test_ans_acc: %.4f" \
    #             % (epoch+1, float(equation_ac)/eval_total, float(value_ac)/eval_total))

        # if epoch == n_epochs - 1:
            # best_acc_fold.append((equation_ac, value_ac, eval_total))
    embedding_scheduler.step()
    encoder_scheduler.step()
    numencoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    decoder_scheduler.step()