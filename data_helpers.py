# Author:BinHaoWang
# System Date:2019/1/3
# System Time:07:35   
import numpy as np
import tensorflow as tf
import re
import itertools
from collections import Counter
import os
import time
import pickle


# 读取文件
def read_and_clean_zh_file(input_file):
    lines = list(open(input_file, "r", encoding='utf-8').readlines())
    lines = [clean_str(seperate_line(line)) for line in lines]

    """ 
    # 这里只是查看处理后的数据【数据被覆盖】
    
    # 这个是为了存储处理后的数据的位置
    output_cleaned_file = os.path.abspath(os.path.join(os.curdir, "text"))
    if not os.path.exists(output_cleaned_file):
        os.makedirs(output_cleaned_file)

    output_cleaned_file = os.path.join(output_cleaned_file, 'output_cleaned_file.txt')

    # 写入处理后的数据
    with open(output_cleaned_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    """
    return lines


# 将字与字之间，以空格隔开 【中国我爱你——> 中 国 我 爱 你】
def seperate_line(line):
    return ''.join([word + ' ' for word in line])


# 处理中文符号
def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_positive_negative_data_files(positive_data_file, negative_data_file):
    """
        从文件中加载MR极性数据，将数据分解为单词并生成标签。
        返回拆分的句子和标签。
    """
    # 加载数据文件
    positive_examples = read_and_clean_zh_file(positive_data_file)
    negative_examples = read_and_clean_zh_file(negative_data_file)

    # 编制资料
    x_text = positive_examples + negative_examples

    # 生成标签
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


# ["讲 的 是 孔 子 后 人", "讲 的 是 孔 子 后 人"] --> ['讲', '的', '是', '孔', '子', '后', '人', '讲', '的', '是', '孔', '子', '后', '人']
# 求最大长度, 如果每行的长度小于最大长度,那就用标识来填充, 如果大于就分割[:max_sentence_length]
def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
        [len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return [sentences, max_sentence_length]


def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        # 将obj对象序列化为字节（bytes）写入到file文件中
        pickle.dump(input_dict, f)


def loadDict(dict_file, output_dict=None):
    with open(dict_file, 'rb') as f:
        # 从一个对象文件中读取序列化数据，将其反序列化之后返回一个对象
        output_dict = pickle.load(f)
    return output_dict


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    为数据集生成批处理迭代器
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            # 在每个时期洗牌数据
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_idx: end_idx]


def load_data_and_labels(input_text_file, input_label_file, num_labels):
    x_text = read_and_clean_zh_file(input_text_file)
    y = None if not os.path.exists(input_label_file) else map(int, list(open(input_label_file, "r", encoding='utf-8').readlines()))
    return [x_text, y]
