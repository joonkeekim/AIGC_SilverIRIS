import re
import json
from konlpy.tag import Kkma
import pickle
import pdb
from pororo import Pororo
# stanza.download('ko')
from tqdm import tqdm
tagger = Kkma()
dp = Pororo(task="dep_parse", lang="ko")
def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {'+': 0, '-': 0, '*': 1, '/': 1, '@': 1, '#': 1, '^': 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res

NUM_PATTERN = re.compile('(\(\d+\.*\d+\/\d+\.*\d+\)|\d+\.\d+|\d+)%?')
# stanza_tokenizer = stanza.Pipeline('ko', processors='tokenize')
def transfer_num(data):  # transfer num into "NUM"
    print("trnasfer number")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    cnt = 0
    for d in tqdm(data):
        idx = d["id"]
        nums = []
        input_seq = []
        text = d["segmented_text"]
        equations = d["equation"]
        ee = equations
        if equations[:2] == 'x=':
          equations = equations[2:]

        input_seq, nums, num_pos, res = tokenize_text(text)
        
        if len(nums) >= 10:
            continue

        if copy_nums < len(nums):
            copy_nums = len(nums)

        num_dict = {num: i for i, num in enumerate(nums)}
        nums_fraction = nums
        equations = preprocess_expr(equations)
        expr_list = parse_expr(equations)
        prefix = []
        for e in expr_list:
          e = e if len(e) == 1 else refine_fraction(e)
          if e in num_dict:
            prefix.append('N' + str(num_dict[e]))
          elif len(e) > 1 and '//' and '/' in e:
            ea, eb = e.split('/')
            if ea in num_dict:
              prefix.append('N' + str(num_dict[ea]))
            else:
              prefix.append(ea)
            prefix.append('/')
            if eb in num_dict:
              prefix.append('N' + str(num_dict[eb]))
            else:
              prefix.append(eb)
          else:
            prefix.append(e)
        # pdb.set_trace()
        out_seq = infix_to_prefix(prefix)
        out_seq2 = from_infix_to_postfix(prefix)
        try:
          # out_seq = infix_to_prefix(prefix)
          infix = prefix_to_infix(out_seq)
          expr = detokenize_expr(infix, nums)
          expr = postprocess_expr(expr)

        except:
          cnt += 1
          continue
          
        # out_seq = prefix_to_infix(out_seq)    
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
        if len(nums) != len(num_pos):
          pdb.set_trace()
        # assert (len(nums) == len(num_pos))
        pairs.append((idx,input_seq, out_seq, out_seq2, nums, num_pos, res))
        # print(pairs)
    
    temp_g = []
    allowed_tokens = ['0', '1', '2', '3', '4', '5,' '6', '7', '8', '9', '10', '12', '100', '1000', '10000' '3.14']
    for g in generate_nums:
        if generate_nums_dict[g] >= 5 and g in allowed_tokens:
            temp_g.append(g)

    # del dp
    print(cnt, 'pairs are ignored')
    return pairs, temp_g, copy_nums

def read_json(filename):
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

def delete_decimal_comma(text):
  while True:
    pos = re.search('\d+\,\d{3}', text)
    if pos:
      tmp = text[:pos.start()]
      tmp += text[pos.start():pos.end()].replace(',', '')
      tmp += text[pos.end():]
      text = tmp
    else:
      break
  return text

def convert_parenthese(text):
  text = text.replace('[', '(')
  text = text.replace(']', ')')
  text = text.replace('（', '(')
  text = text.replace('）', ')')
  return text
  
def delete_useless_parenthese(text):
  pass


def separate_ops(text):
  while True:
    pos = re.search('(\+|\-|\*|\/|\)|\(\@\#)(\+|\-|\*|\/|\)|\(\@\#)', text)
    if pos:
      tmp = text[:pos.start()]
      tmp += text[pos.start()] + ' ' + text[pos.end()-1]
      tmp += text[pos.end():]
      text = tmp
    else:
      break
  return text

def refine_fraction(fraction):
  fraction = fraction.replace('(', '').replace(')', '').strip()
  return fraction

def tokenizers(text):
    # dp = Pororo(task="dep_parse", lang="ko")
    # l = []
    # for sentence in doc.sentences:
        # tmp = [token.text.strip() for token in sentence.tokens]
        # l.extend(tmp)
    res = dp(text)
    return res

def tokenize_text(text):
  num_list = []
  tmp = ''
  while True:
    pos = re.search(NUM_PATTERN, text)
    if pos:
      num = text[pos.start():pos.end()]
      tmp += text[:pos.start()]
      text = text[pos.end():]
      if num == '0':
        tmp += ' ' + num + ' '
      else:
        tmp +=  ' ' + 'NUM' + ' '
        num_list.append(num)
    else:
      tmp += text
      break
  # tokenized_text = [t for t in tmp.split(' ') if t]
  res = tokenizers(tmp)
  tokenized_text = [r[1] for r in res]
  num_index_list = [i for i, text in enumerate(tokenized_text) if text == 'NUM']
  num_list = list(map(refine_fraction, num_list))
  return tokenized_text, num_list, num_index_list, res


def remove_useless_number(text):
  mapping = {'1': '일', '2': '이', '3': '삼', '4': '사', '5': '오', '6': '육', '7': '칠'}
  new_text = ''
  while True:
    pos = re.search('(1|2|3|4|5|6|7)\s*학년', text)
    if pos:
      new_text += text[:pos.start()]
      w = text[pos.start():pos.end()].replace(' ', '')[:-2]
      new_text += (mapping[w] + '학년')
      text = text[pos.end():]
    else:
      new_text += text
      break
  return new_text

def convert_unit(text):
  text = text.replace('위안', '원')
  text = text.replace('퍼센트', '%')
  text = text.replace('미터', 'm')
  text = text.replace('키로', 'kg')
  text = text.replace('킬로미터', 'km')
  return text

def convert_wordnum_to_number(text):
  mapping = {
    '한':'1', '하나':'1', '첫': '1', '첫째': '1',
    '두':'2', '둘': '2', '둘째': '2',
    '세':'3', '셋': '3', '셋째': '3',
    '네':'4', '넷': '4', '넷째': '4',
    '다섯':'5', '다섯째': '5',
    '여섯':'6', '여섯째': '6',
    '일곱':'7', '일곱째': '7',
    '여덟':'8', '여덟째': '8',
    '아홉':'9', '아홉째': '9',
    '열':'10', '열째': '10'}
  infos = tagger.pos(text)
  new_text = ''
  for w, p in infos:
    if p in ['MDN', 'NNG', 'NR', 'NU'] and w in mapping:
      pos = re.search(w, text)
      new_text += text[:pos.start()] + mapping[w] 
      text = text[pos.end():]
    elif p == 'MDT' and w == '첫':
      pos = re.search(w, text)
      new_text += text[:pos.start()] + mapping[w] 
      text = text[pos.end():]
  new_text += text
  return new_text

def preprocess_text(text):
  text = delete_decimal_comma(text)
  text = convert_wordnum_to_number(text)
  text = remove_useless_number(text)
  text = convert_unit(text)
  text = separate_ops(text)
  text = convert_parenthese(text)
#  tt, nl, _ = tokenize_text(text)
#  tt = tagger.morphs(' '.join(tt))
#  for i in range(len(tt)):
#    if tt[i] == 'NUM':
#      tt[i] = nl.pop(0)
#  text = ' '.join(tt)
  return text

def check_fraction(text):
  if re.search('(\d+\.\d+|\d+)\/(\d+\.\d+|\d+)', text):
    return True
  else:
    return False

def preprocess_expr(expr):
  """ Preprocessing `expr`. Use before `parse_expr`.
  """
  expr = expr.replace(' ', '')
  expr = expr.replace('//', '@')
  expr = expr.replace('**', '^')
  tmp = ''
  while True:
    pos = re.search('(\d|\))%(\d|\()', expr)
    if pos:
      tmp += expr[:pos.start()]
      tmp += expr[pos.start():pos.end()].replace('%', '#')
      expr = expr[pos.end():]
    else:
      tmp += expr
      break
  expr = tmp
  return expr

def postprocess_expr(expr):
  """ Postprocessing `expr_infix`"""
  tmp = ''
  while True:
    pos = re.search('(\d+|\d+\.\d+)%', expr)
    if pos:
      tmp += expr[:pos.start()]
      tmp += str(float(expr[pos.start():pos.end()-1]) * 0.01)
      expr = expr[pos.end():]
    else:
      tmp += expr
      break
  expr = tmp
  expr = expr.replace('@', '//')
  expr = expr.replace('^', '**')
  expr = expr.replace('#', '%')
  return expr

def evaluate_py_expr(py_expr, auto_int=True):
  ans = round(eval(py_expr), 2)
  ans = '{:.2f}'.format(ans)
  if auto_int and ans[-2:] == '00':
    ans = str(int(float(ans)))
  return ans


def add_explicit_mul(expr):
  while True:
    pos = re.search('(\d+\(|\)\d+)', expr)
    if pos:
      tmp = expr[:pos.start()]
      tmp += expr[pos.start():pos.end()-1] + '*' + expr[pos.end()-1]
      tmp += expr[pos.end():]
      expr = tmp
    else:
      break
  return expr

def convert_ans(ans):
  ans = add_explicit_mul(ans)
  ans = preprocess_expr(ans)
  ans = convert_to_py_expr(ans)
  ans = evaluate_py_expr(ans)
  return ans

def infix_to_prefix(expr_list):
  priority = {'+': 0, '-': 0, '*': 1, '/': 1, '@': 1, '#': 1, '^': 2}
  tmp = []
  prefix = []
  for e in reversed(expr_list):
    if e == ')':
      tmp.append(e)
    elif e == '(':
      a = tmp.pop()
      while a != ')':
        prefix.append(a)
        a = tmp.pop()
    elif e in priority:
      while len(tmp) > 0 and tmp[-1] != ')' and priority[e] < priority[tmp[-1]]:
        prefix.append(tmp.pop())
      tmp.append(e)
    else:
      prefix.append(e)
  while len(tmp) > 0:
    prefix.append(tmp.pop())
  prefix.reverse()
  return prefix

def prefix_to_infix(expr_list):
  ops = ['+', '-', '*', '/', '^', '@', '#']
  tmp = []
  for e in reversed(expr_list):
    if e not in ops:
      tmp.append(e)
    else:
      new_e = '(' + tmp.pop(-1) + e
      new_e += tmp.pop(-1) + ')'
      tmp.append(new_e)
  infix = ''.join(tmp)
  return infix

def postfix_to_infix(expr_list):
  ops = ['+', '-', '*', '/', '^', '@', '#']
  tmp = []
  for e in (expr_list):
    # pdb.set_trace()
    if e not in ops:
      tmp.insert(0,e)
    else:
      op1 = tmp[0]
      tmp.pop(0)
      op2 = tmp[0]
      tmp.pop(0)
      tmp.insert(0, "(" + op2 + e +
                        op1 + ")")
      # new_e = '(' + tmp.pop(-1) + e
      # new_e += tmp.pop(-1) + ')'
      # tmp.append(new_e)
  postfix = ''.join(tmp)
  return postfix

def parse_expr(expr):
  expr_list = []
  while True:
    pos = re.search(NUM_PATTERN, expr)
    if pos:
      expr_list.extend(expr[:pos.start()])
      num = expr[pos.start():pos.end()]
      if ''.join(expr_list[-2:]) in ['--', '(-']:
        num = '-' + num
        expr_list.pop(-1)
      expr_list.append(num)
      expr = expr[pos.end():]
    else:
      expr_list.extend(expr)
      break
  return expr_list

def filter_with_input_length(pairs):
  cnt = 0
  new_pairs = []
  for pair in pairs:
    if len(pair[1]) <= 40 and len(pair[4]) < 20:
      new_pairs.append(pair)
    else:
      cnt += 1
  print(cnt, 'pairs are ignored by length..')
  return new_pairs

def detokenize_expr(expr, num_list):
  for i, num in enumerate(num_list):
    expr = expr.replace('N{}'.format(i), num)
  return expr

def load_my_data(path):
  with open(path, 'r') as f:
    data = [json.loads(line) for line in f]
  return data

def save_lang(lang, path):
  with open(path, 'wb') as f:
    pickle.dump(lang, f)

def load_lang(path):
  with open(path, 'rb') as f:
    lang = pickle.load(f)
  return lang

def save_generate_nums(generate_nums, path):
  with open(path, 'wb') as f:
    pickle.dump(generate_nums, f)

def load_generate_nums(path):
  with open(path, 'rb') as f:
    generate_nums = pickle.load(f)
  return generate_nums

def save_copy_nums(copy_nums, path):
  with open(path, 'wb') as f:
    pickle.dump(copy_nums, f)

def load_copy_nums(path):
  with open(path, 'rb') as f:
    copy_nums = pickle.load(f)
  return copy_nums

def filter_data(pairs):
  allowed_tokens = \
    ['[', ']', '{', '}', '(', ')', '+', '-', '*', '^', '/', '@', '#'] + \
    ['0', '1', '2', '3', '4', '5,' '6', '7', '8', '9', '10', '12', '100', '1000', '10000' '3.14'] + \
    ['N' + f'{i}' for i in range(11)]
  tmp_pairs = []
  cnt = 0
  for pair in pairs:
    expr = pair[4]
    flag = False
    for e in expr:
      if e not in allowed_tokens:
        # print(e, end='\t')
        flag = True
    if flag:
      # print()
      # print(expr)
      # print(pair[0])
      # print()
      cnt += 1
    else:
      tmp_pairs.append(pair)
  pairs = tmp_pairs
  print(f'{cnt} pairs are ignored')
  return pairs


if __name__ == '__main__':
  data = load_my_data('Math_23K_papago_agc.json')
  for d in data:
    text = d['segmented_text']
    id_ = d['id']
    ttext = preprocess_text(text)
    if check_fraction(ttext):
      print(id_)
      print(text)
      print(ttext)
      print()