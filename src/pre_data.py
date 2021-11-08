import random
import json
import copy
import re
import numpy as np
import pdb

PAD_token = 0


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i
    
    def build_input_lang_for_pos(self):
        self.index2word = ["PAD", "UNK"] + self.index2word
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


def transfer_num2(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        idx = d["id"]
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            elif s != "":
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((idx, input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def texts_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(word)
        else:
            res.append("UNK")
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def num_list_processed(num_list):
    st = []
    for p in num_list:
        pos1 = re.search("\d+\(", p)
        pos2 = re.search("\)\d+", p)
        if pos1:
            st.append(eval(p[pos1.start(): pos1.end() - 1] + "+" + p[pos1.end() - 1:]))
        elif pos2:
            st.append(eval(p[:pos2.start() + 1] + "+" + p[pos2.start() + 1: pos2.end()]))
        elif p[-1] == "%":
            st.append(float(p[:-1]) / 100)
        else:
            st.append(eval(p))
    return st


def num_order_processed(num_list):
    num_order = []
    num_array = np.asarray(num_list)
    for num in num_array:
        num_order.append(sum(num>num_array)+1)
    
    return num_order


def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums):
    input1_lang = Lang()
    input2_lang = Lang()
    output1_lang = Lang()
    output2_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        if pair[-1]:
            input1_lang.add_sen_to_vocab(pair[1])
            input2_lang.add_sen_to_vocab(pair[2])
            output1_lang.add_sen_to_vocab(pair[4])
            output2_lang.add_sen_to_vocab(pair[5])
    
    input1_lang.build_input_lang(trim_min_count)
    input2_lang.build_input_lang_for_pos()
    output1_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    output2_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[4]:
            temp_num = []
            flag_not = True
            if word not in output1_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[6]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[6]))])
        try:
            num_stack.reverse()
            input1_cell = indexes_from_sentence(input1_lang, pair[1])
            texts_cell = texts_from_sentence(input1_lang, pair[1])
            input2_cell = indexes_from_sentence(input2_lang, pair[2])
            output1_cell = indexes_from_sentence(output1_lang, pair[4], True)
            output2_cell = indexes_from_sentence(output2_lang, pair[5], False)
            num_list = num_list_processed(pair[6])
            num_order = num_order_processed(num_list)
            train_pairs.append((pair[0], texts_cell, input1_cell, input2_cell, pair[3], len(input1_cell), 
                                output1_cell, len(output1_cell), output2_cell, len(output2_cell), 
                                pair[6], pair[7], num_stack, num_order))
        except:
            continue
    print('Indexed %d words in input language, %d words in output1, %d words in output2' % 
          (input1_lang.n_words, output1_lang.n_words, output2_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        # temp pairs 0 : id , 1: text 2 : pos 3 : parse. 4 : out_seq 5 : out_seq2 6 : num  7 :numpos
        num_stack = []
        for word in pair[4]:
            temp_num = []
            flag_not = True
            if word not in output1_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[6]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[6]))])
        try:
            num_stack.reverse()
            input1_cell = indexes_from_sentence(input1_lang, pair[1])
            texts_cell = texts_from_sentence(input1_lang, pair[1])
            input2_cell = indexes_from_sentence(input2_lang, pair[2])
            output1_cell = indexes_from_sentence(output1_lang, pair[4], True)
            output2_cell = indexes_from_sentence(output2_lang, pair[5], False)
            num_list = num_list_processed(pair[6])
            num_order = num_order_processed(num_list)
            test_pairs.append((pair[0], texts_cell, input1_cell, input2_cell, pair[3], len(input1_cell), 
                            output1_cell, len(output1_cell), output2_cell, len(output2_cell), 
                            pair[6], pair[7], num_stack, num_order))
        except:
            continue
    print('Number of testind data %d' % (len(test_pairs)))
    return input1_lang, input2_lang, output1_lang, output2_lang, train_pairs, test_pairs


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq


# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    id_batches = []
    input_lengths = []
    output1_lengths = []
    output2_lengths = []
    nums_batches = []
    batches = []
    input1_texts = []
    input1_batches = []
    input2_batches = []
    output1_batches = []
    output2_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_order_batches = []
    num_size_batches = []
    parse_graph_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[5], reverse=True)
        input_length = []
        output1_length = []
        output2_length = []
        for _, _, _, _, _, i, _, j,_, k, _, _, _, _ in batch:
            input_length.append(i)
            output1_length.append(j)
            output2_length.append(k)
        input_lengths.append(input_length)
        output1_lengths.append(output1_length)
        output2_lengths.append(output2_length)
        input_len_max = input_length[0]
        output1_len_max = max(output1_length)
        output2_len_max = max(output2_length)
        
        id_batch = []
        input1_text = []
        input1_batch = []
        input2_batch = []
        output1_batch = []
        output2_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_order_batch = []
        num_size_batch = []
        parse_tree_batch = []
        for idx, text, i1, i2, parse_tree, li, j, lj, k, lk, num, num_pos, num_stack, num_order in batch:
            id_batch.append(idx)
            input1_text.append(text)
            input1_batch.append(pad_seq(i1, li, input_len_max))
            input2_batch.append(pad_seq(i2, li, input_len_max))
            output1_batch.append(pad_seq(j, lj, output1_len_max))
            output2_batch.append(pad_seq(k, lk, output2_len_max))
            num_batch.append(len(num))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_order_batch.append(num_order)
            num_size_batch.append(len(num_pos))
            parse_tree_batch.append(parse_tree)
        
        id_batches.append(id_batch)
        input1_texts.append(input1_text)
        input1_batches.append(input1_batch)
        input2_batches.append(input2_batch)
        output1_batches.append(output1_batch)
        output2_batches.append(output2_batch)
        nums_batches.append(num_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_order_batches.append(num_order_batch)
        num_size_batches.append(num_size_batch)
        parse_graph_batches.append(get_parse_graph_batch(input_length, parse_tree_batch))
        
    return id_batches, input1_texts, input1_batches, input2_batches, input_lengths, output1_batches, output1_lengths, output2_batches, output2_lengths, \
           nums_batches, num_stack_batches, num_pos_batches, num_order_batches, num_size_batches, parse_graph_batches


def get_parse_graph_batch(input_length, parse_tree_batch):
    # pdb.set_trace()
    batch_graph = []
    max_len = max(input_length)
    for i in range(len(input_length)):
        parse_tree = parse_tree_batch[i]
        diag_ele = [1] * input_length[i] + [0] * (max_len - input_length[i])
        graph1 = np.diag([1]*max_len) + np.diag(diag_ele[1:], 1) + np.diag(diag_ele[1:], -1)
        graph2 = copy.deepcopy(graph1)
        graph3 = copy.deepcopy(graph1)
        for j in range(len(parse_tree)):
            if parse_tree[j] != -1:
                try:
                    graph1[j, parse_tree[j]] = 1
                except:
                    pdb.set_trace()
                graph2[parse_tree[j], j] = 1
                graph3[j, parse_tree[j]] = 1
                graph3[parse_tree[j], j] = 1
        graph = [graph1.tolist(), graph2.tolist(), graph3.tolist()]
        batch_graph.append(graph)
    batch_graph = np.array(batch_graph)
    return batch_graph


def word2vec(train_pairs, embedding_size, input_lang):
    sentences = []
    for train in train_pairs:
        sentence = train[1]
        sentences.append(sentence)
    
    from gensim.models import word2vec
    model = word2vec.Word2Vec(sentences, size=embedding_size, min_count=1)

    emb_vectors = []
    emb_vectors.append(np.zeros((embedding_size)))
    for i in range(1, input_lang.n_words):
        emb_vectors.append(np.array(model.wv[input_lang.index2word[i]]))
    
    return emb_vectors