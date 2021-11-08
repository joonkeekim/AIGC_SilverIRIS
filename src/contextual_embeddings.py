import torch.nn as nn
import torch
# from .tokenization_kobert import KoBertTokenizer
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from transformers import ElectraTokenizerFast, ElectraModel, AutoTokenizer
import pdb
from typing import Tuple, List

def resize_outputs(
        outputs: torch.Tensor, bpe_head_mask: torch.Tensor, bpe_tail_mask: torch.Tensor, max_word_length: int
) -> Tuple[torch.Tensor, List]:
    """Resize output of pre-trained transformers (bsz, max_token_length, hidden_dim) to word-level outputs (bsz, max_word_length, hidden_dim*2). """
    batch_size, input_size, hidden_size = outputs.size()
    word_outputs = torch.zeros(batch_size, max_word_length, hidden_size).to(outputs.device)
    sent_len = list()

    for batch_id in range(batch_size):
        head_ids = [i for i, token in enumerate(bpe_head_mask[batch_id]) if token == 1]
        tail_ids = [i for i, token in enumerate(bpe_tail_mask[batch_id]) if token == 1]
        assert len(head_ids) == len(tail_ids)

        # word_outputs[batch_id][0] = torch.cat(
        #     (outputs[batch_id][0], outputs[batch_id][0])
        # )  # replace root with [CLS]
        for i, (head, tail) in enumerate(zip(head_ids, tail_ids)):
            word_outputs[batch_id][i] = torch.mean(torch.stack((outputs[batch_id][head], outputs[batch_id][tail])),dim=0)
        sent_len.append(i)

    return word_outputs, sent_len


class BertEncoder(nn.Module):
    def __init__(self, bert_model='klue/roberta-base', device='cuda:0', freeze_bert=False):
        super(BertEncoder, self).__init__()
        print(bert_model)
        if bert_model == 'pretrained_mlm_nsp_roberta':
            self.bert_layer = RobertaModel.from_pretrained('klue/roberta-large')  # pytorch
            self.bert_tokenizer = AutoTokenizer.from_pretrained('pretrained_mlm_nsp')
        else:
            self.bert_layer = RobertaModel.from_pretrained(bert_model)  # pytorch
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.device = device
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

    def get_bpe_masks(self, sentences):
        max_length = 128
        bpe_heads = []
        bpe_tails = []

        for sentence in sentences:
            # pdb.set_trace()
            bpe_head_mask = [0]
            bpe_tail_mask = [0]
            # token_list = sentence.split()
            token_list = sentence
            # pdb.set_trace()
            for token in token_list:
                bpe_len = len(self.bert_tokenizer.tokenize(token))
                head_token_mask = [1] + [0] * (bpe_len - 1)
                tail_token_mask = [0] * (bpe_len - 1) + [1]
                bpe_head_mask.extend(head_token_mask)
                bpe_tail_mask.extend(tail_token_mask)



            bpe_head_mask.append(0)
            bpe_tail_mask.append(0)
            if len(bpe_head_mask) > max_length:
                bpe_head_mask = bpe_head_mask[:max_length]
                bpe_tail_mask = bpe_tail_mask[:max_length]

            else:
                bpe_head_mask.extend([0] * (max_length - len(bpe_head_mask)))  # padding by max_len
                bpe_tail_mask.extend([0] * (max_length - len(bpe_tail_mask)))  # padding by max_len
            bpe_heads.append(bpe_head_mask)
            bpe_tails.append(bpe_tail_mask)

        all_bpe_heads = torch.tensor([b for b in bpe_heads], dtype=torch.long)
        all_bpe_tails = torch.tensor([b for b in bpe_tails], dtype=torch.long)
        return all_bpe_heads, all_bpe_tails

    def bertify_input(self, sentences):
        '''
        Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids
        '''
        # Tokenize the input sentences for feeding into BERT
        # pdb.set_trace()
        all_tokens = [['[CLS]'] + self.bert_tokenizer.tokenize(" ".join(sentence)) + ['[SEP]'] for sentence in sentences]
        bpe_head_masks, bpe_tail_masks = self.get_bpe_masks(sentences)
        # Pad all the sentences to a maximum length
        input_lengths = [len(tokens) for tokens in all_tokens]
        max_length = max(input_lengths)
        padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

        # Convert tokens to token ids
        token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(
            self.device)

        # Obtain attention masks
        pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
        attn_masks = (token_ids != pad_token).long()

        return token_ids, attn_masks, input_lengths, bpe_head_masks, bpe_tail_masks

    def forward(self, sentences, max_len):
        '''
        Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
        '''
        # Preprocess sentences
        # print(sentences)
        token_ids, attn_masks, input_lengths, bpe_head_masks, bpe_tail_masks = self.bertify_input(sentences)
        # Feed through bert
        cont_reps = self.bert_layer(token_ids, attention_mask=attn_masks)
        cont_reps = cont_reps.last_hidden_state
        # max_word_length = max(torch.sum(bpe_head_masks, dim=1)).item()
        # pdb.set_trace()
        word_outputs, sent_len = resize_outputs(cont_reps, bpe_head_masks, bpe_tail_masks, max_len)
        return word_outputs, token_ids


class RobertaEncoder(nn.Module):
    def __init__(self, roberta_model='roberta-base', device='cuda:0 ', freeze_roberta=False):
        super(RobertaEncoder, self).__init__()
        self.roberta_layer = RobertaModel.from_pretrained(roberta_model)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
        self.device = device

        if freeze_roberta:
            for p in self.roberta_layer.parameters():
                p.requires_grad = False

    def robertify_input(self, sentences):
        '''
        Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids

        '''
        # Tokenize the input sentences for feeding into RoBERTa
        all_tokens = [['<s>'] + self.roberta_tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]

        # Pad all the sentences to a maximum length
        input_lengths = [len(tokens) for tokens in all_tokens]
        max_length = max(input_lengths)
        padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

        # Convert tokens to token ids
        token_ids = torch.tensor([self.roberta_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(
            self.device)

        # Obtain attention masks
        pad_token = self.roberta_tokenizer.convert_tokens_to_ids('<pad>')
        attn_masks = (token_ids != pad_token).long()

        return token_ids, attn_masks, input_lengths

    def forward(self, sentences):
        '''
        Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token
        '''
        # Preprocess sentences
        token_ids, attn_masks, input_lengths = self.robertify_input(sentences)

        # Feed through RoBERTa
        cont_reps, _ = self.roberta_layer(token_ids, attention_mask=attn_masks)

        return cont_reps, input_lengths


if __name__ == '__main__':
    b = BertEncoder(device='cpu')
    # res = b.bertify_input(['나는 밥을 먹었다.', '나는 밥을 안 먹었다.'])
    res = b.forward(['나는 밥을 먹었다.', '나는 밥을 안 먹었다.'])
    print(res)
    print(b.bert_tokenizer.decode(2))
    # pdb.set_trace()
