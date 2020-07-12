# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
# 项目解读
# https://blog.csdn.net/zengNLP/article/details/102635246
# https://mp.weixin.qq.com/s/_xILvfEMx3URcB-5C8vfTw
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


# def build_vocab(file_path, tokenizer, max_size, min_freq):
#     vocab_dic = {}
#     with open(file_path, 'r', encoding='UTF-8') as f:
#         for line in tqdm(f):
#             lin = line.strip()
#             if not lin:
#                 continue
#             content = lin.split('\t')[0]
#             for word in tokenizer(content):
#                 vocab_dic[word] = vocab_dic.get(word, 0) + 1
#         vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
#         vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
#         vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
#     return vocab_dic

def build_vocab(file_path, tokenizer, max_size, min_freq):
    #词/字典: key=单字, val=字频
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f): #遍历每一行
            lin = line.strip() #去掉首尾空白符
            if not lin: #遇到空行 跳过
                continue
            content = lin.split('\t')[0] #text  label；每一行以\t为切分，拿到文本
            for word in tokenizer(content): #分词 or 分字
                vocab_dic[word] = vocab_dic.get(word, 0) + 1 #构建词或字到频数的映射 统计词频/字频【dict.get(key1, key2): 若key1不存在，则获取key2】
        # ❤根据 【min_freq【过滤】低频词】，并按频数从大到小排序，然后取前max_size个单词（dict.items() -- 得到key, val）
        # ❤得到的vocab_list是一个按词频降序的list(已过滤低频字),【list元素为 dict.items() —> i.e. tuple(字, 词频)！！！！】
        vocab_listTuple = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        #构建词或字到索引的映射 从0开始             word_with_freq: tuple(字，词频)
        vocab_dic = {word_with_freq[0]: idx for idx, word_with_freq in enumerate(vocab_listTuple)}
        #添加未知符和填充符的映射
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def build_dataset(config, ues_word):
    # 定义tokenizer函数（word-level/character-level）
    if ues_word:  # 基于词 提前用分词工具把文本分开 以空格为间隔
        tokenizer = lambda x: x.split(' ')  # 直接以空格分开 word-level
    else:  # 基于字符  <--本项目中没有结合词组，只是单词级别，∴数据集小，训练快
        tokenizer = lambda x: [y for y in x]  # char-level

    # 构建词/字典
    if os.path.exists(config.vocab_path):  # 如果存在构建好的词/字典 则加载
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:  # 构建词/字典（基于训练集）
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 保存构建好的词/字典
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    # 词/字典大小
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):  # 遍历每一行  # tqdm：进度条显示
                lin = line.strip()  # 去掉首尾空白符
                if not lin:  # 遇到空行 跳过
                    continue
                content, label = lin.split('\t')  # text  label；每一行以\t为切分
                words_line = []
                token = tokenizer(content)  # 对文本进行分词/分字  # 获取单字列表(lambda将一句话的content转化为单字list)
                seq_len = len(token)  # 序列/文本真实长度（填充或截断前）
                if pad_size:  # 长截短填
                    if len(token) < pad_size:  # 文本真实长度比填充长度短 # 当前句长 < 指定时，将填充特定字符(PAD), 只为补齐序列（统一格式）
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))  # 填充
                    else:  # 文本真实长度比填充长度 长
                        token = token[:pad_size]  # 截断
                        seq_len = pad_size  # 把文本截断后(指定的最大长度)设置为填充长度
                # word to id （找到单字在vocab中的idx，再依据idx找到对应的词向量）
                for word in token:  # 将词/字转换为索引，dict.get(key1, key2)【key1不在词/字典中的 用key2: UNK对应的索引代替】
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], label,seq_len),...]

    # 分别对训练集、验证集、测试集进行处理 把文本中的词或字转换为词/字典中的索引
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    # 返回字/词典 预处理好的训练集、验证集、测试集
    return vocab, train, dev, test


class DatasetIterater(object): # 传入的dataset就是batches
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device) # data
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device) # label

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):

        if self.residue and self.index == self.n_batches: # batch数量为整数 且 最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    # 预训练词/字向量路径
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    # 嵌入维度
    emb_dim = 300
    # 词/字嵌入矩阵存储路径
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"

    if os.path.exists(vocab_dir):  # 如果有处理好的词/字典
        word_to_id = pkl.load(open(vocab_dir, 'rb'))  # 直接读取 词/字到索引的映射
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        # 构建词/字典
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 保存词/字典
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)  # 随机初始化词/字嵌入矩阵 # 设一句话序列最长300

    # 读取预训练词/字向量
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):  # 遍历每一行 格式：词/字 300个数字(均以空格分开)
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        # 用预训练词/字向量覆盖 随机初始化的词/字嵌入矩阵
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()

    # 保存初始化的词/字嵌入矩阵
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

#%% Test build_vocab()中的vocab_list、vocab_dic
dict = {'a': 1, 'b':20, 'c':12, 'd':52, 'w':33}
sorted_list = sorted([_ for _ in dict.items() if _[1]>=10], key=lambda x: x[1], reverse=True) # [('d', 52), ('w', 33), ...]
print([(idx, cnt) for idx, cnt in enumerate(sorted_list)])  # [(0, ('d', 52)), (1, ('w', 33)), (2, ('b', 20)), (3, ('c', 12))]