#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
parser_utils.py: Utilities for training the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
"""

import time
import os
import logging
from collections import Counter
#用于统计词频的类，避免使用for来计数
from . general_utils import get_minibatches
from parser_transitions import minibatch_parse

from tqdm import tqdm
import torch
import numpy as np

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'


class Config(object):#存储基础设置
    language = 'english'
    with_punct = True
    unlabeled = True#是否要考虑dependency label，不仅仅是指向
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    def __init__(self, dataset):#training_set
        #初始化，Parser类大量数据成员,根据语料库建立token（单词，pos label，dependency label）附上id，
        #同时不忘了未知（unk）、补位空（NULL）以及根（ROOT）。  
    
        #ex是字典，h与l是分别来词头和标签，找由root发出的标签。head=0，label按理说是root
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        #counter用于统计词频，返回的是包含很多词的字典
        if len(counter) > 1:#检查整个语料库root label一共有几种
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]#返回root标签
        
        #################################################
        #1. tok2id引入dependency label
        
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))
        #w是语料库中每一句话中每个词的dependency label；
        #set去重，剩下的是这句中出现的其他label；deprel就是这句中所有label
        #形成每个dependency label的编号的字典，key是<l>+label名称
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)#tok2id中<l><NULL>记录的是dependency label的数量
        #将config中的默认参数带进来，作为其内部的数据成员
        config = Config()
        self.unlabeled = config.unlabeled
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep
        self.language = config.language

        if self.unlabeled:#根据是否考虑标签生成trans的种类
            trans = ['L', 'R', 'S']
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)
        
        self.n_trans = len(trans)#trans的数量，并将每种trans转换为id，id转换为trans
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = {i: t for (i, t) in enumerate(trans)}
        
        #################################################
        #2. tok2id引入pos label
        
        # dict.update()函数主要用于更新字典，可增加可改动，但不可删除
        # logging.info('Build dictionary for part-of-speech tags.')
        # build_dict返回 pos的key:index_value。
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                  offset=len(tok2id)))
        #字典中引入了下面三个key，并赋予了值为整体的长度，分别代表pos中未知的，空的与根，相当于在当前基础上往后顺三个index
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        #################################################
        #3. tok2id引入word label
        
        # logging.info('Build dictionary for words.')
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                  offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)
        
        #生成tok2id的同时，又生成id2tok，双向查询
        self.tok2id = tok2id
        #.items()就是将字典转换为list of tuple，经常在for循环中使用
        self.id2tok = {v: k for (k, v) in tok2id.items()}
        
        #特征数和neural based dependency parser一致
        self.n_features = 18 + (18 if config.use_pos else 0) + (12 if config.use_dep else 0)
        #n_tokens表示
        self.n_tokens = len(tok2id)

    def vectorize(self, examples):
        # train_set = parser.vectorize(train_set)
        #这里的example是个list of dicts
        vec_examples = []
        for ex in examples:#每个句子循环
            #列表生成式，for...if的话，不可以在if后加else；但如果if在前的话，必须加else，即if...else...for
            #下面就是将句子转换为index序列，如果UNK和P_UNK代表未知的word或者pos，未知有相应的index
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]#每一句形成[len(token)最终长度,句子单词1的index，单词2的index,...]
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]#每一句形成[len(token)引入pos和dependency后的长度,句子单词1的pos_index，单词2的pos_index,...]
            head = [-1] + ex['head'] #head本身就有序号，但那个是在句子中的序号
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]#未知的dependency label的index为-1，就是最后一个，毕竟上文没有定义相关的
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        #vec_examples装的就是每个单词，其词性，其dependency label的index；head还是原来的序号，单词在句子中的，而不是tok2id中的
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        # 调用：self.extract_features(stack, buf, arcs, ex)，stack，buffer都是当前状态
        #arc是(被依赖的单词，依赖的单词，gold_t)
        #主要用于根据当前的状态，提取特征
        if stack[0] == "ROOT":
            stack[0] = 0

        def get_lc(k):#依赖于k的词，且在k左边的
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):#依赖于k的词，且在k右边的
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        p_features = []
        l_features = []
        ###########################3
        # 引入词的特征：stack和buffer顶上三个词
        
        #[ex['word'][x] for x in stack[-3:]]这个是堆顶三个单词的index；前半部分是如果不存在的用null的index去补充
        features = [self.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]
        #buffer前三个词的index+缺少的拿null的index去补充
        features += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3 - len(buf))
        
        ##########################
        # 引入pos特征，也就是对应上那六个词的特征
        if self.use_pos:
            p_features = [self.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]]
            p_features += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3 - len(buf))
        ############################
        # 引入stack顶前两个词的 lc1 lc2 rc1 rc2；llc rrc
        for i in range(2):#两个词
            if i < len(stack):
                k = stack[-i-1]
                lc = get_lc(k)
                rc = get_rc(k)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep:
                    l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)
            else:
                features += [self.NULL] * 6
                if self.use_pos:
                    p_features += [self.P_NULL] * 6
                if self.use_dep:
                    l_features += [self.L_NULL] * 6

        features += p_features + l_features
        assert len(features) == self.n_features
        return features

    def get_oracle(self, stack, buf, ex):#stack,buf,ex三个参数分别是[单词index]
        if len(stack) < 2:#stack如果元素比较少，直接返回self.n_trains -1=2(unlabeled)
            return self.n_trans - 1

        i0 = stack[-1]#堆顶
        i1 = stack[-2]#堆次顶
        h0 = ex['head'][i0]
        h1 = ex['head'][i1]
        l0 = ex['label'][i0]
        l1 = ex['label'][i1]

        if self.unlabeled:#不算dependency label
            if (i1 > 0) and (h1 == i0):#i1不为root，且由i0指向i1，i1依靠于i0
                return 0#0代表着left-arc
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):#i1只要存在就行，由i1指向i0，没有buffer中任何元素依附于i0
                return 1#1代表着right-arc
            else:#如果buf为空，返回None代表着结束，否则返回2代表shift
                return None if len(buf) == 0 else 2
        #
        else:#包括dependency label
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        #调用示例：parser.create_instances(train_set)
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            n_words = len(ex['word']) - 1#词数-1是因为有root

            # arcs = {(h, t, label)}
            #下面的各个列表中装了句子单词的序号
            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []#表示句内词之间的 指向关系
            instances = []
            #下面的注释都以unlabeled为例
            for i in range(n_words * 2):
                ###################################3
                # 这他妈gold_t应该是y，就是我们要预测的真值！
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:#buffer为空
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                #去提取特征，除了extract_features生成的，还包括，legal_labels和gold_t
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))
                #shift
                if gold_t == self.n_trans - 1:
                    stack.append(buf[0])
                    buf = buf[1:]
                #right-arc 移走堆顶第二个
                elif gold_t < self.n_deprel: 
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack = stack[:-2] + [stack[-1]]
                else:#left-arc 移走堆顶第一个
                    arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))
                    stack = stack[:-1]
            #for...else...语法，for执行完了之后执行else
            else:
                succ += 1
                all_instances += instances

        return all_instances

    def legal_labels(self, stack, buf):#根据stack和buffer的状态生成legal_labels
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel#unlabeled时，n_deprel=1
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        return labels#labels的size是1*(2*n_deprel+1)

    def parse(self, dataset, eval_batch_size=5000):
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(dataset):
            #每句话进行循环
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, dataset, sentence_id_to_idx)
        dependencies = minibatch_parse(sentences, model, eval_batch_size)

        UAS = all_tokens = 0.0
        with tqdm(total=len(dataset)) as prog:
            for i, ex in enumerate(dataset):
                head = [-1] * len(ex['word'])
                for h, t, in dependencies[i]:
                    head[t] = h
                for pred_h, gold_h, gold_l, pos in \
                        zip(head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                        assert self.id2tok[pos].startswith(P_PREFIX)
                        pos_str = self.id2tok[pos][len(P_PREFIX):]
                        if (self.with_punct) or (not punct(self.language, pos_str)):
                            UAS += 1 if pred_h == gold_h else 0
                            all_tokens += 1
                prog.update(i + 1)
        UAS /= all_tokens
        return UAS, dependencies


class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx

    def predict(self, partial_parses):
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies,
                                             self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')
        mb_x = torch.from_numpy(mb_x).long()
        mb_l = [self.parser.legal_labels(p.stack, p.buffer) for p in partial_parses]

        pred = self.parser.model(mb_x)
        pred = pred.detach().numpy()
        pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'), 1)
        pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in pred]
        return pred


def read_conll(in_file, lowercase=False, max_example=None):#读取数据的函数，专门读取conll文件
#conll文件是用于存储经过标记的句子集，而存在的格式，以空行来区分不同句子，空格来区分不同列，每一行代表一个单词
#https://blog.csdn.net/gammag/article/details/78523053 解释的非常好
#本文的格式为：
#0 id索引
#1 word或者是标点
#2 词形的词条或词干
#3 google提取的通用POS
#4 语言特定的词性标签，下划线为不可用
#5 特征列表，下划线为不可用
#6 指向当前词dependencies的发出者的编号(0代表root)，表示依赖于...，
#7 该词与head的依赖关系种类
#8 二级依赖列表
#9 其他注释
    examples = []#字典列表，每个句子是一个字典
    # with ... as ... 在读取文件上用的很多，用于替代try...except...finally...
    # as的作用在于，等价于f=open(in_file)
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        #word主要用于装单词
        #pos词性
        #head表示依赖于
        #label代表依赖关系种类
        for line in f.readlines():
            sp = line.strip().split('\t')#区分出每行的各列内容
            if len(sp) == 10:#判断是否为空行，空行是区分句子的方式
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])#根据要去转换大小写
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:#为空行，且积累了一些单词了（用于防止连续空多行），记录到example里
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    #在本例中这个没有，都是None；max_example应该用于限制句子例子的数量
                    break
        if len(word) > 0:#最后一行收个尾
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return examples


def build_dict(keys, n_max=None, offset=0):#建立一个字典
    #build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],offset=len(tok2id))，
    #这里的keys是'<p>词性'，offset是原来的字典序号到多少了
    count = Counter()
    for key in keys:
        count[key] += 1#对每种key计数
    #直接count=Counter(keys)即可
    #创建的dict中是否要限制'n_max'，即词形的数量（只挑选最多的几种）
    #ls是词形key：计数value的字典
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)
    #这里的return的是什么？字典 词性：index
    ##！！这里注意 count.most_common返回的可是list of tuples
    #Counter({2: 3, 1: 2, 3: 1})-->[(2, 3), (1, 2), (3, 1)]
    #所以w[0]就是Count里的key
    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        # http://nlp.stanford.edu/software/spanish-faq.shtml
        return pos in ["f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh",
                       "fia", "fit", "fp", "fpa", "fpt", "fs", "ft",
                       "fx", "fz"]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)


def minibatches(data, batch_size):
    #后文的调用：minibatches(train_data, batch_size)
    x = np.array([d[0] for d in data])#这个就是features
    y = np.array([d[2] for d in data])####这就是gold_t
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1#y是one-hot的形式，这部分开始不再考虑dependency label
    return get_minibatches([x, one_hot], batch_size)


def load_and_preprocess_data(reduced=True):#读取和预处理写的很棒，按函数来的
    #reduced 控制什么数据集，大/小
    config = Config()

    print("Loading data...",)
    start = time.time()
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    #trained_set dev_ser test_set三个字典列表
    if reduced:
        #大数据集，1000句训练集，500验证集，500测试集
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Building parser...",)
    start = time.time()
    parser = Parser(train_set)#利用training_set构建一个Parser对象，里面装的最重要的是tok2id，trans2id等等，
    print("took {:.2f} seconds".format(time.time() - start))

    print("Loading pretrained embeddings...",)
    start = time.time()
    word_vectors = {}
    for line in open(config.embedding_file).readlines():#一行一行读取
        #文件第一列(index=0)为单词，剩下的为embedding
        sp = line.strip().split()
        #将嵌入的每一项转化为float
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    #embedding_matrix用于做什么？？size是（token的数量，50：词向量维度数），为啥要正态分布一下呢？
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')
    #对字典进行for in实际上是对字典里的key进行循环
    for token in parser.tok2id:
        i = parser.tok2id[token] #token对应id
        if token in word_vectors:#对应的是token中的单词（还有其他的dependency label和pos label）
            embeddings_matrix[i] = word_vectors[token]
        #以上，把语料库种所有的单词的嵌入保存在embedding_matrix中，其余dependency label和pos label需要在训练中调整，所以上文要进行正态初始化
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Vectorizing data...",)
    #向量化，把dataset转换为 list of dict [{'word': word, 'pos': pos,'head': head, 'label': label}]
    #这里word都是把每个词转化为tok2id的id
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...",)
    start = time.time()
    train_examples = parser.create_instances(train_set)
    #a list of tuples (self.extract_features(stack, buf, arcs, ex), legal_labels, gold_t)
    print("took {:.2f} seconds".format(time.time() - start))

    return parser, embeddings_matrix, train_examples, dev_set, test_set,


class AverageMeter(object):
    """Computes and stores the average and current value"""
    #单纯用于计算平均值
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    pass
