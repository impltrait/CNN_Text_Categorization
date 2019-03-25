# Author:BinHaoWang
# System Date:2019/1/3
# System Time:07:37   
import os
import sys
import logging
import multiprocessing
import time
import json
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def embedding_sentences(sentences, embedding_size=128, window=5, min_count=5, file_to_load=None, file_to_save=None):

    # 加载模型
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        """
        Word2Vec 模型
        size参数 向量的维数 主要是用来设置神经网络的层数，Word2Vec 中的默认值是设置为100层。更大的层次设置意味着更多的输入数据，不过也能提升整体的准确度，合理的设置范围为 10~数百
        min_count 忽略总频率低于这个值的所有单词 在不同大小的语料集中，我们对于基准词频的需求也是不一样的。譬如在较大的语料集中，
            我们希望忽略那些只出现过一两次的单词，这里我们就可以通过设置min_count参数进行控制。一般而言，合理的参数值会设置在0~100之间
        workers: 用于设置并发训练时候的线程数，不过仅当Cython安装的情况下才会起作用：
        """
        w2vModel = Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count,
                            workers=multiprocessing.cpu_count())
        # 判断保存Word2Vec模型的路径是否存在
        if file_to_save is not None:
            # 保存
            w2vModel.save(file_to_save)

    all_vectors = []

    # 特征向量的维数 128
    embeddingDim = w2vModel.vector_size

    """
        embeddingUnknown: [0, 0, 0, 0, 0,······,0, 0, 0, 0, 0]是128维的向量
    """
    embeddingUnknown = [0 for i in range(embeddingDim)]

    """
      sentences[0]:  ['讲', '的', '是', '孔',  '回', '到', '家', '乡', ······, '<PADDING>'] 是190维的向量，也就是最大长度
    """
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            """
               w2vModel.wv.vocab: 词表
               word: 是一个字
               w2vModel[word]:[-0.13266018 -0.25414982  0.28642225 ······ 0.17015015  0.2760758 ] 是128维的向量

            """
            # 判断某个词是否在词表中, 如果在, 用w2vModel[word]得到所对应的128维的向量
            # 如果不在, 用embeddingUnknown这个向量替代这个字
            # 在追加在事先定义的 this_vector 列表中
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        """
        this_vector: 
            [
                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ],  这个长度是128
                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ],
                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ]
            ]
        this_vector长度是190
        """
        all_vectors.append(this_vector)
        """
            len(all_vectors):10001
            all_vectors: [
            
                            10001个:[
                                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ],
                                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ],
                                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ]
                            ]
                            ·
                            ·
                            ·
                            [
                                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ],
                                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ],
                                array([ 0.46655697,  0.20291726, -1.0600653 ····· 0.04125202,  0.08269487, -1.2300228 ]
                            ]
                        ]
        """
    return all_vectors


