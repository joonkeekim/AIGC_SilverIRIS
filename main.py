import pdb
import os
import time
import torch.optim
import torch.nn as nn
import random
import kss
import numpy as np
from datasets import load_dataset
from src.logger import *
from src.models import *
from src.train_and_evaluate import *
from src.text_utils import *
from src.expressions_transfer import *
from src.contextual_embeddings import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering,Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "true"
problem_path = '/home/agc2021/dataset/problemsheet_5_00.json' # submit
# problem_path = 'for_testing/problemsheet_integegrated.json'
answer_path = 'answersheet_5_00_eparkskkuedu.json'
embedding_size = 1024
hidden_size = 512
beam_size = 5
n_layers = 2
hop_size = 2
emb_name = 'pretrained_mlm_nsp'
device = 'cuda:0'
freeze_emb = False

def read_sheet(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        problems = json.load(f)
    return problems

def evaluate_py_expr(py_expr, auto_int=True):
    is_neg = False
    if eval(py_expr) < 0:
        py_expr = '-('+py_expr+')'
        is_neg = True
    ans = round(eval(py_expr), 2)
    ans = '{:.2f}'.format(ans)
    if auto_int and ans[-2:] == '00':
        ans = str(int(float(ans)))
    return ans, is_neg

########### cls
cls_tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
cls_model = AutoModelForSequenceClassification.from_pretrained('cls_model',num_labels=2).to(device)

def get_prediction(text):
    target_names = ['tree','qa']
    # prepare our text into tokenized sequence
    max_length = 256
    inputs = cls_tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    inputs.pop('token_type_ids')
    outputs = cls_model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax()]

############# qa

qa_tokenizer = AutoTokenizer.from_pretrained("jkjk_qa")
qa_model = AutoModelForQuestionAnswering.from_pretrained("jkjk_qa")
processed_path = "./processed_qa_data.json"
max_length = 384 
doc_stride = 128 

def split_context_question(text):
    splited_text = kss.split_sentences(text)
    context_list = splited_text[0:-1]
    context=""
    for text in context_list:
        context += text + " "
    question = splited_text[-1]
    if context == "":
        context = question
    return context, question


def data_preprocess(text, key=1):
    final_output = {}
    new_data = []
    context, question = split_context_question(text)
    context = context.rstrip()
    new_data.append({"context": context, "question": question, "id": key})
    final_output["data"] = new_data
    with open("./processed_qa_data.json", "w") as f:
        json.dump(final_output, f, indent=2)

def prepare_validation_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = qa_tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    tokenized_examples.pop('token_type_ids')

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def qa_infer(idx, output, validation_features, x_valid):
    n_best_size = 20
    max_answer_length = 30
    try:
        start_logits = output.start_logits[idx].cpu().numpy()
        end_logits = output.end_logits[idx].cpu().numpy()
        offset_mapping = validation_features[idx]["offset_mapping"]
        context = x_valid[idx]["context"]
    
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                if start_index <= end_index:
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char],
                            "start": start_char,
                            "end": end_char
                        }
                    )

        valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
        ans_txt = valid_answers[0]["text"]
        answer = ans_txt.split(',')[0]
        answer = answer.rstrip()
        answer = answer.split(' ')[0]
        answer = answer.rstrip()
        _len = len(answer)
        start_idx = valid_answers[0]["start"]
        end_idx = start_idx + _len

        return answer, start_idx, end_idx
    except:
        return "error", 0, 1

def qa(text, k):
    result_data = dict()
    data_preprocess(text, k)
    testset = load_dataset('json', data_files=processed_path, field="data")
    x_valid = testset["train"]
    validation_features = x_valid.map(prepare_validation_features,batched=True,remove_columns=x_valid.column_names)
    trainer = Trainer(qa_model, eval_dataset=validation_features,tokenizer=qa_tokenizer )

    for batch in trainer.get_eval_dataloader():
        break
    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
    with torch.no_grad():
        output = trainer.model(**batch)

    valid_answers, start, end = qa_infer(0, output, validation_features, x_valid)
    # _id = x_valid[0]["id"]
    # result_data[_id] = {"answer": valid_answers, "equation": f"print('{text}'[{start}:{end}])"}
    os.remove("./processed_qa_data.json")

    # print(result_data)
    return valid_answers, f"print('{text}'[{start}:{end}])" 

############ alge
def alge_init(pickle_path, model_path, model_type):
    input1_lang = load_lang(pickle_path+'input1_lang.pickle')
    input2_lang = load_lang(pickle_path+'input2_lang.pickle')
    output1_lang = load_lang(pickle_path+'output1_lang.pickle')
    output2_lang = load_lang(pickle_path+'output2_lang.pickle')
    # train_pairs = load_lang('pickles/train_pairs.pickle')
    # test_pairs = load_lang('pickles/test_pairs.pickle')
    generate_nums = load_lang(pickle_path+'generate_nums.pickle')
    copy_nums = load_lang(pickle_path+'copy_nums.pickle')

    generate_num1_ids = []
    generate_num2_ids = []
    for num in generate_nums:
        generate_num1_ids.append(output1_lang.word2index[num])
        generate_num2_ids.append(output2_lang.word2index[num])

    embedding = BertEncoder(emb_name, device, freeze_emb, model_path[-5:-1])
    encoder = EncoderSeq(input1_size=input1_lang.n_words, input2_size=input2_lang.n_words,
                         embedding1_size=embedding_size, embedding2_size=embedding_size // 4,
                         hidden_size=hidden_size, n_layers=n_layers, hop_size=hop_size)
    numencoder = NumEncoder(node_dim=hidden_size, hop_size=hop_size)

    embedding.to(device)
    encoder.to(device)
    numencoder.to(device)
    embedding.load_state_dict(torch.load(
        model_path+'embedding.pt', map_location=device))
    encoder.load_state_dict(torch.load(
        model_path+'encoder.pt', map_location=device))
    numencoder.load_state_dict(torch.load(
        model_path+'numencoder.pt', map_location=device))

    if model_type == 'ALL':
        predict = Prediction(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                             input_size=len(generate_nums))
        generate = GenerateNode(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                                embedding_size=embedding_size)
        merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
        decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size,
                                 input_size=output2_lang.n_words, output_size=output2_lang.n_words, n_layers=n_layers)

        predict.to(device)
        generate.to(device)
        merge.to(device)
        decoder.to(device)

        predict.load_state_dict(torch.load(
            model_path+'predict.pt', map_location=device))
        generate.load_state_dict(torch.load(
            model_path+'generate.pt', map_location=device))
        merge.load_state_dict(torch.load(
            model_path+'merge.pt', map_location=device))
        decoder.load_state_dict(torch.load(
            model_path+'decoder.pt', map_location=device))

    if model_type == 'TREE':
        predict = Prediction(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                             input_size=len(generate_nums))
        generate = GenerateNode(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                                embedding_size=embedding_size)
        merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

        predict.to(device)
        generate.to(device)
        merge.to(device)

        predict.load_state_dict(torch.load(
            model_path+'predict.pt', map_location=device))
        generate.load_state_dict(torch.load(
            model_path+'generate.pt', map_location=device))
        merge.load_state_dict(torch.load(
            model_path+'merge.pt', map_location=device))

        decoder = None

    if model_type == 'GRU':
        decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size,
                                 input_size=output2_lang.n_words, output_size=output2_lang.n_words, n_layers=n_layers)
        decoder.to(device)

        decoder.load_state_dict(torch.load(
            model_path+'decoder.pt', map_location=device))

        predict = None
        generate = None
        merge = None
    return [input1_lang, input2_lang, output1_lang, output2_lang, generate_num1_ids, generate_num2_ids, generate_nums,embedding, encoder, numencoder, predict, generate, merge, decoder]


def alge_infer(input_seq, nums, num_pos, res, is_trimed, model_type, input1_lang, input2_lang, output1_lang, output2_lang, generate_num1_ids, generate_num2_ids,generate_nums,
               embedding, encoder, numencoder, predict, generate, merge, decoder):
    try:
        arcs = [r[2] for r in res]
        parse_tree = [arc - 1 if arc != -1 else arc for arc in arcs]
        pos = [r[3] for r in res]
        num_stack = []

        num_stack.reverse()
        input1_cell = indexes_from_sentence(input1_lang, input_seq)
        if is_trimed:
            texts_cell = texts_from_sentence(input1_lang, input_seq)
        else:
            texts_cell = input_seq
        input2_cell = indexes_from_sentence(input2_lang, pos)
        num_list = num_list_processed(nums)
        num_order = num_order_processed(num_list)

        parse_graph = get_parse_graph_batch([len(input1_cell)], [parse_tree])
        # pdb.set_trace()
        if model_type == "ALL":
            result_type, test_res, score = evaluate_double(texts_cell, input1_cell, input2_cell, len(input1_cell),
                                                           generate_num1_ids, generate_num2_ids,
                                                           embedding, encoder, numencoder, predict, generate, merge, decoder,
                                                           input1_lang, output1_lang, output2_lang, num_pos, num_order,
                                                           parse_graph, beam_size=5, device=device)
        if model_type == "TREE":
            result_type, test_res, score = evaluate_tree(texts_cell, input1_cell, input2_cell, len(input1_cell),
                                                         generate_num1_ids, generate_num2_ids,
                                                         embedding, encoder, numencoder, predict, generate, merge, decoder,
                                                         input1_lang, output1_lang, output2_lang, num_pos, num_order,
                                                         parse_graph, beam_size=5, device=device)
        if model_type == "GRU":
            result_type, test_res, score = evaluate_gru(texts_cell, input1_cell, input2_cell, len(input1_cell),
                                                        generate_num1_ids, generate_num2_ids,
                                                        embedding, encoder, numencoder, predict, generate, merge, decoder,
                                                        input1_lang, output1_lang, output2_lang, num_pos, num_order,
                                                        parse_graph, beam_size=5, device=device)

        results = []
        # num_list
        num_list = [str(n) for n in num_list]
        # num_list = []
        # pdb.set_trace()
        if result_type == "tree":
            try:
                result = out_expression_list(test_res, output1_lang, num_list)
                if result is None:
                    # pdb.set_trace()
                    return random.choice(generate_nums if len(num_list) == 0 else num_list), num_list, -100
                try:
                    result = prefix_to_infix(result)
                except:
                    return random.choice(generate_nums if len(num_list) == 0 else num_list), num_list, -100
            except:
                return random.choice(generate_nums if len(num_list) == 0 else num_list), num_list, -100
        else:
            if test_res[-1] == output2_lang.word2index["EOS"]:
                test_res = test_res[:-1]
            try:
                result = out_expression_list(test_res, output2_lang, num_list)
                if result is None:
                    # pdb.set_trace()
                    return random.choice(generate_nums if len(num_list) == 0 else num_list), num_list, -100
                try:
                    result = postfix_to_infix(result)
                except:
                    return random.choice(generate_nums if len(num_list) == 0 else num_list), num_list, -100
            except:
                return random.choice(generate_nums if len(num_list) == 0 else num_list), num_list, -100
        return result, num_list, score
    except:
        num_list = []
        return random.choice(generate_nums if len(num_list) == 0 else num_list), num_list, -100

pure = alge_init(pickle_path='pickles_new/',model_path='weights/pure/',model_type='ALL')
pure2 = alge_init(pickle_path='pickles_new/',model_path='weights/2pure/',model_type='ALL')
total = alge_init(pickle_path='pickles/',model_path='weights/all/',model_type='ALL')
total2 = alge_init(pickle_path='pickles/',model_path='weights/all2/',model_type='ALL')
tree = alge_init(pickle_path='pickles/',model_path='weights/tree/',model_type='TREE')
gru = alge_init(pickle_path='pickles/',model_path='weights/gru/',model_type='GRU')


def alge(text):
    text = preprocess_text(text)
    input_seq, nums, num_pos, res = tokenize_text(text)
    # 방법은 4개
    result_score_dict = {}
    results = []
    # pure lm
    with torch.no_grad():
        result, num_list, score = alge_infer(input_seq, nums, num_pos, res, False, "ALL", *pure)
        result_score_dict['0'] = score
        results.append([result,num_list])
        # trim all
        result, num_list, score = alge_infer(input_seq, nums, num_pos, res,True, "ALL",*total)
        result_score_dict['1'] = score
        results.append([result,num_list])
        # # trim tree
        result, num_list, score = alge_infer(input_seq, nums, num_pos, res,True, "TREE",*tree)
        result_score_dict['2'] = score
        results.append([result,num_list])
        # # trim gru
        result, num_list, score = alge_infer(input_seq, nums, num_pos, res,True, "GRU",*gru)
        result_score_dict['3'] = score
        results.append([result,num_list])
        # pdb.set_trace()
        result, num_list, score = alge_infer(input_seq, nums, num_pos, res,True, "ALL",*total2)
        result_score_dict['4'] = score
        results.append([result,num_list])
        #
        result, num_list, score = alge_infer(input_seq, nums, num_pos, res, False, "ALL", *pure2)
        result_score_dict['5'] = score
        results.append([result,num_list])

        final = max(result_score_dict,key=result_score_dict.get)
        result, num_list = results[int(final)]

        try:
            pred_ans, is_neg = evaluate_py_expr(result)
            if is_neg:
                result = '-('+result+')'
                # print(input_seq)
        except:
            if num_list != 0:
                pred_ans = random.choice(num_list)
            else:
                pred_ans = random.choice([1,2,3,4,5,6,7,8,9,10,11,12])
    # print(results,result_score_dict,sep='\n\n')
    return pred_ans, result

def do_infer():
    test_data = read_sheet(problem_path)
    # dp = Pororo(task="dep_parse", lang="ko")
    result_data = dict()
    r,w=0,0
    for k, v in test_data.items():
        text = v['question']
        
        # classification

        cls = get_prediction(text)
        if cls == 'tree':
            pred_ans, result = alge(text)
            result_data[k] = {
                'answer': pred_ans,
                'equation': 'print(' + result + ')'
            }
        else:
            pred_ans, result = qa(text, k)
        # pdb.set_trace()
            result_data[k] = {
                'answer': pred_ans,
                'equation': result
            }
        # print(pred_ans)
        ###
    # print(f'right : {r} ,wrong : {w}')
    with open(answer_path, 'w') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    do_infer()