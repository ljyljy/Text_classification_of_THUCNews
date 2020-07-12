# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    # 定义tokenizer函数（word-level/character-level）
    if ues_word:  # 基于词 提前用分词工具把文本分开 以空格为间隔
        tokenizer = lambda x: x.split(' ')  # 直接以空格分开 word-level
    else:  # 基于字符
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

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):  # 遍历每一行
                lin = line.strip()  # 去掉首尾空白符
                if not lin:  # 遇到空行 跳过
                    continue
                content, label = lin.split('\t')  # text  label；每一行以\t为切分
                words_line = []
                token = tokenizer(content)  # 对文本进行分词/分字
                seq_len = len(token)  # 序列/文本真实长度（填充或截断前）

                if pad_size:  # 长截短填
                    if len(token) < pad_size:  # 文本真实长度比填充长度 短
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                    else:  # 文本真实长度比填充长度 长
                        token = token[:pad_size]
                        seq_len = pad_size  # 把文本真实长度设置为填充长度
                # word to id
                for word in token:  # 将词/字转换为索引，不在词/字典中的 用UNK对应的索引代替
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # fasttext ngram
                buckets = config.n_gram_vocab
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(pad_size):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                # -----------------
                contents.append((words_line, int(label), seq_len, bigram, trigram))
        return contents  # [([...], labels,seq_len,[bigram],[trigram]),  ...]

    # 分别对训练集、验证集、测试集进行处理
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    # 返回字/词典 预处理好的训练集、验证集、测试集
    return vocab, train, dev, test


class DatasetIterater(object):  # 自定义数据集迭代器
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # 构建好的数据集
        self.n_batches = len(batches) // batch_size  # 得到batch数量
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:  # 不能整除
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # 转换为tensor 并 to(device)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  # 原始文本对应的索引
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)  # 标签
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)  # bigram对应的索引
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)  # trigram对应的索引

        # seq_len为文本的实际长度（不包含填充的长度） 转换为tensor 并 to(device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:  # 当数据集大小 不整除 batch_size时，构建最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)  # 把最后一个batch转换为tensor 并 to(device)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:  # 构建每一个batch
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)  # 把当前batch转换为tensor 并 to(device)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:  # 不能整除
            return self.n_batches + 1  # batch数+1
        else:
            return self.n_batches


def build_iterator(dataset, config):  # 构建数据集迭代器
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == "__main__":
    '''提取预训练词向量'''
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/vocab.embedding.sougou"
    word_to_id = pkl.load(open(vocab_dir, 'rb'))
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
