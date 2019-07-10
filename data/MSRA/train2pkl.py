#coding:utf-8
import codecs
import re
import pandas as pd
import numpy as np

# def wordtag():
#     with codecs.open('train.txt','r','utf-8') as input_data:
#         with codecs.open('wordtag.txt','w','utf-8') as output_data:
#             for line in input_data.readlines():
#                 # 去掉每行头尾空格，并以""做分隔成一个列表
#                 line = line.strip().split()
#                 if len(line) == 0:
#                     continue
#                 # print(line)
#                 # 针对每一行数据每一个元素遍历做处理
#                 for word in line:
#                     word = word.split('/')
#                     # print(word)
#                     if word[1]!='o':
#                         # 实体由一个字符构成
#                         if len(word[0]) == 1:
#                             output_data.write(word[0] + "/B_" + word[1] + " ")
#                         # 实体由两个字符构成
#                         elif len(word[0]) == 2:
#                             output_data.write(word[0][0] + "/B_" + word[1] + " ")
#                             output_data.write(word[0][1] + "/E_" + word[1] + " ")
#                         # 实体由三个及三个以上字符构成
#                         else:
#                             output_data.write(word[0][0] + "/B_" + word[1] + " ")
#                             for j in word[0][1:len(word[0]) - 1]:
#                                 output_data.write(j + "M_" + word[1] + " ")
#                             output_data.write(word[0][-1] + "/E_" + word[1] + " ")
#                     else:
#                         for j in word[0]:
#                             output_data.write(j + "/o" + " ")
#                 output_data.write('\n')
#
# wordtag()
datas = list()
labels = list()
linedata = list()
linelabel = list()

tag2id = {
    '' :0,
    'B_ns' :1,
    'B_nr' :2,
    'B_nt' :3,
    'M_nt' :4,
    'M_nr' :5,
    'M_ns' :6,
    'E_nt' :7,
    'E_nr' :8,
    'E_ns' :9,
    'o': 0
}

id2tag = {
    0:'' ,
    1:'B_ns' ,
    2:'B_nr' ,
    3:'B_nt' ,
    4:'M_nt' ,
    5:'M_nr' ,
    6:'M_ns' ,
    7:'E_nt' ,
    8:'E_nr' ,
    9:'E_ns' ,
    10: 'o'
}

with codecs.open('wordtag.txt','r','utf-8') as input_data:
    for line in input_data.readlines():
        line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
        for sen in line:
            sen = sen.strip().split()
            if len(sen) == 0:
                continue
            linedata = []
            linelabel = []
            num_not_o = 0
            for word in sen:
                word = word.split('/')
                linedata.append(word[0])
                linelabel.append(tag2id[word[1]])

                if word[1] != 'o':
                    num_not_o += 1
            if num_not_o != 0:
                datas.append(linedata)
                labels.append(linelabel)


print(len(datas))
print(len(labels))
# print(datas) # [['把', '欧', '美'], ['港', '台', '流', '行', '的', '食', '品', '类', '图', '谱']]
# print(labels) # [[0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]

# itertools.chain是平坦任何嵌套的可迭代级别的最佳解决方案，它具有很高的效率
import collections
def flat_gen(x):
    def iselement(e):
        return not(isinstance(e, collections.Iterable) and not isinstance(e, str))
    for el in x:
        if iselement(el):
            yield el
        else:
            yield from flat_gen(el)

all_words = flat_gen(datas)
# print(all_words)

sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
# print(sr_allwords)

set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1)
# print(set_words)
# print(set_ids)

word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
# print(word2id)
# print(id2word)


word2id["unknow"] = len(word2id)+1
# print(word2id)

max_len = 50
def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

def y_padding(ids):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
df_data['x'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
x = np.asarray(list(df_data['x'].values))
y = np.asarray(list(df_data['y'].values))

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.2, random_state=43)


print('Finished creating the data generator.')
import pickle
import os
with open('../dataMSRA.pkl', 'wb') as outp:
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
    pickle.dump(x_train, outp)
    pickle.dump(y_train, outp)
    pickle.dump(x_test, outp)
    pickle.dump(y_test, outp)
    pickle.dump(x_valid, outp)
    pickle.dump(y_valid, outp)
print('Finished saving the data.')