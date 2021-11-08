from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, BertForNextSentencePrediction
from transformers import ElectraTokenizerFast, ElectraModel, AutoTokenizer
import torch
import json

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import random
import torch.nn.functional as F
import pickle
import torch.nn as nn
from torch.utils.data.dataset import random_split
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
DEVICE = 'cuda:0'

tok = AutoTokenizer.from_pretrained('/home/ubuntu/joonkee/pretraining/tokenizer_base')

with open('pretrain_data/transfered_data.pkl', 'rb') as f:
    data = pickle.load(f)


class BERTLanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, sep_id: str='[SEP]', cls_id: str='[CLS]',
                mask_id: str='[MASK]', pad_id: str="[PAD]", seq_len: int=256, mask_frac: float=0.15, p: float=0.5):
        """Initiate language modeling dataset.
        Arguments:
            data (list): a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab (sentencepiece.SentencePieceProcessor): Vocabulary object used for dataset.
            p (float): probability for NSP. defaut 0.5
        """
        super(BERTLanguageModelingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.seq_len = seq_len
        self.sep_id = tokenizer.sep_token_id
        self.cls_id = tokenizer.cls_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.p = p
        self.mask_frac = mask_frac
        self.mlm_probability = mask_frac

    def __getitem__(self, i):
        seq1 = self.data[i]['segmented_text']
        seq2_idx = i
        
        # decide wheter use random next sentence or not for NSP task
        if random.random() > self.p: # 틀린 데이터
            is_next = torch.tensor(1)
            if 'wrong' in self.data[i].keys():
                rand_idx = random.randint(0,len(self.data[i]['wrong']))
                if rand_idx:
                    seq2 = self.data[seq2_idx]['wrong'][rand_idx-1]
                else:
                    while seq2_idx == i:
                        seq2_idx = random.randint(0, len(data)-1)
                    seq2 = self.data[seq2_idx]['equation']
            else:
                while seq2_idx == i:
                    seq2_idx = random.randint(0, len(data)-1)
                seq2 = self.data[seq2_idx]['equation']
                # seq2 = self.data[seq2_idx]['equation']
            # while seq2_idx == i:
                # seq2_idx = random.randint(0, len(data))

        else: # 맞는 데이터 
            is_next = torch.tensor(0)
            if 'right' in self.data[i].keys():
                rand_idx = random.randint(0,len(self.data[i]['right']))
                if rand_idx:
                    seq2 = self.data[seq2_idx]['right'][rand_idx-1]
                else:
                    seq2 = self.data[seq2_idx]['equation']
            else:
                seq2 = self.data[seq2_idx]['equation']

        # print(seq2)
        encoded_dict = tok.encode_plus(
                        [seq1, seq2],                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = self.seq_len,           # Pad & truncate all sentences.
                        padding='max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
        labels = encoded_dict['input_ids'].clone()
        inputs = encoded_dict['input_ids'].clone()
        # print(labels.shape)
        # print(self.data[i]['id'], i)
        special_tokens_mask = [tok.get_special_tokens_mask(val,already_has_special_tokens=True) for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask,dtype=torch.bool)
        # print(special_token_mask)
        # print(labels)
        # special_tokens_mask = batch.pop("special_tokens_mask", None)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        mlm_train = inputs
        mlm_target = labels
        attn_masks = encoded_dict['attention_mask']
        token_type_ids = encoded_dict['token_type_ids']
        # print(mlm_train[:,:50], mlm_target[:,:50],sep='\n')
        # mlm_train, mlm_target, sentence embedding, NSP target
        # print(mlm_train.shape, mlm_target.shape, attn_masks.shape, token_type_ids.shape, is_next.shape)
        if mlm_train.shape[0] > 256:
            print('?')
        return mlm_train.squeeze(0), mlm_target.squeeze(0), attn_masks.squeeze(0), is_next
        # return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x



class MLM_NSP(nn.Module):
    def __init__(self, voc_size:int=30000):
        super(MLM_NSP, self).__init__()
        d_model = 1024
        # intermediate_hidden = 3072
        self.linear_mlm1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-12)
        self.linear_mlm2 = nn.Linear(d_model, voc_size)

        self.linear_nsp1 = nn.Linear(d_model, d_model)
        self.act2 = nn.Tanh()
        self.linear_nsp2 = nn.Linear(d_model, 2)

    def forward(self, input_seq):
        '''
        param:
            input: a batch of sequences of words
            seg: Segmentation embedding for input tokens
        dim:
            input:
                input: [B, S]
                seg: [B, S]
            output:
                result: [B, S, V]
        '''

        output_mlm = self.linear_mlm1(input_seq) # [B, S, voc_size]
        output_mlm = self.act(output_mlm) # [B, S, voc_size]
        output_mlm = self.layer_norm(output_mlm) # [B, S, voc_size]
        output_mlm = self.linear_mlm2(output_mlm) # [B, S, voc_size]

        output_nsp = self.linear_nsp1(input_seq[:,0,:])
        output_nsp = self.act2(output_nsp)
        output_nsp = self.linear_nsp2(output_nsp)
        # return output_nsp
        return output_mlm, output_nsp

class MyModel(nn.Module):
    def __init__(self, voc_size, pretrained_path):
        super(MyModel, self).__init__()
        d_model = 1024
        # intermediate_hidden = 3072
        self.lm_model = RobertaModel.from_pretrained(pretrained_path)
        self.mlm_nsp_model = MLM_NSP(voc_size)
    def forward(self, mlm_train, attention_mask):
        '''
        param:
            input: a batch of sequences of words
            seg: Segmentation embedding for input tokens
        dim:
            input:
                input: [B, S]
                seg: [B, S]
            output:
                result: [B, S, V]
        '''
        output = self.lm_model(mlm_train, attention_mask=attention_mask).last_hidden_state
        output = self.mlm_nsp_model(output)
        return output

def accuracy(log_pred, y_true):
    y_pred = torch.argmax(log_pred, dim=1).to(y_true.device)
    return (y_pred == y_true).to(torch.float)

def train(model, dataloader, optimizer,valid_loader, total_leng, early_stop_cnt, scheduler, min_loss):
    

    mlm_epoch_loss = 0
    nsp_epoch_loss = 0
    # min_loss = 100
    cnt = 0 # count length for avg loss
    # early_stop_cnt = 0
    stop = False
    for batch, (mlm_train, mlm_target, attn_masks, is_next) in enumerate(tqdm(dataloader)):
        # print(cnt)
        # MLM task
        model.train()
        optimizer.zero_grad()
        ###
        # pdb.set_trace()
        # elec_output = lm_model(mlm_train.to(DEVICE), attention_mask=attn_masks.to(DEVICE)).last_hidden_state
        # output = mlm_nsp_model(elec_output)
        # output_nsp = mlm_nsp_model(elec_output.to(DEVICE))
        ###
        output = model(mlm_train.to(DEVICE), attention_mask=attn_masks.to(DEVICE))
        mlm_output = output[0].reshape(-1, output[0].shape[-1])
        mlm_loss = criterion(mlm_output, mlm_target.to(DEVICE).reshape(-1)) # CE
        nsp_loss = criterion(output[1], is_next.to(DEVICE)) # no need for reshape target
        loss = mlm_loss+nsp_loss
        # loss = nsp_loss
        # torch.nn.utils.clip_grad_norm_(_loss.parameters(), 1)
        loss.backward()
        optimizer.step()
        # NSP tasks

        mlm_epoch_loss += mlm_loss.item()
        nsp_epoch_loss += nsp_loss.item()
        # mlm_loss = 0
        cnt += 1
        if cnt % 20 == 0:
            nsp_acc = accuracy(output[1], is_next).mean()
            with open('log_mlm_nsp.txt', 'a') as f:
                f.write(f'train : {cnt} step,  mlm : {mlm_loss.item():.2f}, nsp : {nsp_loss.item():.2f} nsp_acc : {nsp_acc:.2f}\n')
                
            # print(f'train : {cnt} step,  mlm : {mlm_loss:.2f}, nsp : {nsp_loss.item():.2f} nsp_acc : {nsp_acc:.2f}\n')
        if cnt % 300 == 0:
            mlm, nsp, acc = valid(model, valid_loader, total_leng)
            # with open('log_rein.txt', 'a') as f:
            #     f.write(f'validation : {cnt} step, early_stop_cnt{early_stop_cnt}, mlm : {mlm:.2f}, nsp : {nsp:.2f} nsp_acc : {acc:.2f}\n')
            # print(f'validation : {cnt} step, {early_stop_cnt}, mlm : {mlm:.2f}, nsp : {nsp:.2f} nsp_acc : {acc:.2f}\n')

            if mlm+nsp<min_loss:
                early_stop_cnt = 0
                min_loss = mlm+nsp
                # print('min loss:',min_loss)
                with open('log_mlm_nsp.txt', 'a') as f:
                    f.write('save,,,\n')
                model.module.lm_model.save_pretrained('pretrained_mlm_nsp')
            else:
                early_stop_cnt += 1
            with open('log_mlm_nsp.txt', 'a') as f:
                f.write(f'validation : {cnt} step, early_stop_cnt{early_stop_cnt}, mlm : {mlm:.2f}, nsp : {nsp:.2f} nsp_acc : {acc:.2f} min_loss : {min_loss:.2f}\n')
            # print(f'validation : {cnt} step, {early_stop_cnt}, mlm : {mlm:.2f}, nsp : {nsp:.2f} nsp_acc : {acc:.2f} min_loss : {min_loss:.2f}\n')
            if early_stop_cnt > 10:
                stop = True
                break
        scheduler.step()
    return mlm_epoch_loss / cnt, nsp_epoch_loss / cnt, stop, early_stop_cnt, min_loss

def valid(model, dataloader, total_leng):
    model.eval()
    mlm_epoch_loss = 0
    nsp_epoch_loss = 0
    nsp_acc = 0
    with torch.no_grad():
        cnt = 0 # count length for avg loss
        for batch, (mlm_train, mlm_target, attn_masks, is_next) in enumerate(dataloader):
            # MLM task
            ###
            # elec_output = lm_model(mlm_train.to(DEVICE), attention_mask=attn_masks.to(DEVICE)).last_hidden_state
            # output = mlm_nsp_model(elec_output)
            # output_nsp = mlm_nsp_model(elec_output.to(DEVICE))
            ###
            # pdb.set_trace()
            output = model(mlm_train.to(DEVICE), attention_mask=attn_masks.to(DEVICE))
            mlm_output = output[0].reshape(-1, output[0].shape[-1])
            mlm_loss = criterion(mlm_output, mlm_target.to(DEVICE).reshape(-1)) # CE
            nsp_loss = criterion(output[1], is_next.to(DEVICE)) # no need for reshape target

            # loss = mlm_loss+nsp_loss

            mlm_epoch_loss += mlm_loss.item()
            nsp_epoch_loss += nsp_loss.item()
            cnt += 1
            nsp_acc += accuracy(output[1], is_next).sum()
                # step_nsp_acc = nsp_acc.mean()
            #     with open('log_rein.txt', 'a') as f:
            #         f.write(f'validation!! : {cnt} step,  mlm : {mlm_loss.item():.2f}, nsp : {nsp_loss.item():.2f} nsp_acc : {step_nsp_acc}\n')
            #     print(f'validation : {cnt} step,  mlm : {mlm_loss.item():.2f}, nsp : {nsp_loss.item():.2f} nsp_acc : {step_nsp_acc}\n')

    return mlm_epoch_loss / cnt, nsp_epoch_loss / cnt, nsp_acc/total_leng

dataset = BERTLanguageModelingDataset(data=data,tokenizer=tok)

model = MyModel(tok.vocab_size,'klue/roberta-large')
# model.load_state_dict(torch.load('/home/ubuntu/joonkee/pretraining/pretrained_lm_mlm'))
model.lm_model.from_pretrained('/home/ubuntu/joonkee/pretraining/pretrained_lm_mlm')
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss().cuda()
model.cuda()
print(' ')

train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.95),len(dataset)- int(len(dataset)*0.95)])
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_dataset,batch_size=32, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=3e-5)
optimizer_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-7)

import time
N_EPOCHS = 30
early_stop_cnt = 0
min_loss = 100
for epoch in range(1, N_EPOCHS+1):
    start_time = time.time()
    mlm_loss, nsp_loss, stop, early_stop_cnt, min_loss = train(model, dataloader, optimizer, valid_loader,len(dataset)- int(len(dataset)*0.95), early_stop_cnt,optimizer_scheduler, min_loss)
    end_time = time.time()
    if stop == True:
        break
    print('!!!!!',epoch, mlm_loss, nsp_loss, min_loss ,end_time-start_time)