# Author:BinHaoWang
# System Date:2019/1/3
# System Time:15:08   
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    用于文本分类的CNN
    使用嵌入层，然后是卷积层、最大池层和softmax层。
    """

    def __init__(self, sequence_length, num_classes, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):
        # 输入、池化等占位符
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prod = tf.placeholder(tf.float32, name="dropout_keep_prod")
        # l2正则化损耗(可选)
        l2_loss = tf.constant(0.0)

        # 嵌入层
        """
        self.input_x.shape:(?, 190, 128)
        self.embedded_chars_expended.shape:(?, 190, 128, 1)
        expand_dims(input, axis):在axis轴处给input增加一个为1的维度。 
        """
        self.embedded_chars = self.input_x
        self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)

        # 为每个过滤器大小创建卷积+ maxpool层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope(f'conv-maxpool-{filter_size}'):
                # 卷积层
                """
                filter_shape：
                    [3, 128, 1, 128]
                    [4, 128, 1, 128]
                    [5, 128, 1, 128]
                """
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='weight')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bias")

                conv = tf.nn.conv2d(self.embedded_chars_expended,
                                    w,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv'
                                    )
                # 应用非线性
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # 输出的最大池
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool'
                                        )
                pooled_outputs.append(pooled)

        # 组合所有池功能
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 添加池化
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prod)

        # 最终(非规范化)分数和预测
        with tf.name_scope('output'):
            w = tf.get_variable('w',
                                shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer()
                                )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], name='bias'))
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # 计算平均交叉熵损失
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 精准度
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
