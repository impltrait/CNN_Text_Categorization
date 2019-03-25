# Author:BinHaoWang
# System Date:2019/1/2
# System Time:20:44
import tensorflow as tf

import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers
from text_CNN import TextCNN

# 数据加载参数
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "用于数据分割训练集/测试集的百分比")
tf.flags.DEFINE_string("positive_data_file", "./data/ham_5000.utf8", "正数据")
tf.flags.DEFINE_string("negative_data_file", "./data/spam_5000.utf8", "负数据")
tf.flags.DEFINE_integer("num_labels", 2, "数据的标签数量。(默认值:2)")

# 模型超参数
tf.flags.DEFINE_integer("embedding_dim", 128, "字符嵌入维数(默认:128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "过滤器大小(默认为'3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "每个过滤器大小的过滤器数量(默认值:128)")
tf.flags.DEFINE_float("dropout_keep_prod", 0.5, "下采样概率(默认:0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2正则化lambda(默认值:0.0)")

# 训练参数
tf.flags.DEFINE_integer("batch_size", 64, "批处理大小(默认:64)")
tf.flags.DEFINE_integer("num_epochs", 200, "训练时迭代数(默认值:200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "这许多步骤之后开发集上的Evalue模型(默认:100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "在这个步骤之后保存模型(defult: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "需要存储的检查点数量(默认为5)")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "允许设备软性设备放置")
tf.flags.DEFINE_boolean("log_device_placement", False, "操作系统在设备上的日志位置")

# FLAGS保存命令行参数的数据
FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags() 将其解析成字典存储到FLAGS.__flags中
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

# 为模型和摘要准备输出目录
# os.curdir 当前目录
# os.path.abspath 绝对路径
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.curdir, "runs", timestamp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#  数据预处理
# ================

# 加载数据
x_text, y = data_helpers.load_positive_negative_data_files(FLAGS.positive_data_file, FLAGS.negative_data_file)
"""
y:
[[0 1]
 [0 1]
 [0 1]
 ...
 [1 0]
 [1 0]
 [1 0]]
 
 y.shape: (10001, 2)
"""
# 得到嵌入向量
# 重要：为了使得filter的矩阵的宽度一直，获取最大文档的length,不够长的文档（邮件）用padding方式补充处理
# 用padding处理不够长的内容（一篇邮件）用0填充，保持所有的文档一致
sentences, max_document_length = data_helpers.padding_sentences(x_text, '<PADDING>')
x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size=FLAGS.embedding_dim,
                                                  file_to_save=os.path.join(out_dir, 'trained_word2vec.model')))
"""
x：
[
  [[-0.73885745 -0.4614228  -0.33116117 ... -0.64979428 -0.54315078
    0.3412481 ]
  [-0.85366857 -1.04219687 -1.47315824 ... -0.99841827 -0.22177659
    0.81408805]
  [ 0.07334232 -0.94928586 -2.16250372 ...  1.51250672  0.10350526
   -0.78161925]
  ...
  [ 0.64990038  0.89552945  0.60671443 ...  0.34627447 -0.2319406
    1.26994538]
  [ 0.64990038  0.89552945  0.60671443 ...  0.34627447 -0.2319406
    1.26994538]
  [ 0.64990038  0.89552945  0.60671443 ...  0.34627447 -0.2319406
    1.26994538]]
 ...

 [[-2.47282648  0.47700468 -1.22464252 ... -0.8082239  -1.89889967
    1.11561406]
  [ 0.4014017  -1.64889693  1.2426033  ... -0.37758014 -1.53680599
   -0.89380378]
  [-2.54294491  0.691055   -0.55590552 ... -0.81213897  0.07968619
    2.08492327]
  ...
  [ 0.64990038  0.89552945  0.60671443 ...  0.34627447 -0.2319406
    1.26994538]
  [ 0.64990038  0.89552945  0.60671443 ...  0.34627447 -0.2319406
    1.26994538]
  [ 0.64990038  0.89552945  0.60671443 ...  0.34627447 -0.2319406
    1.26994538]]
]

x.shape: (10001, 190, 128)
"""
# 保存参数
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels': FLAGS.num_labels, 'max_document_length': max_document_length}
data_helpers.saveDict(params, training_params_file)

# 数据随机洗牌
np.random.seed(666)
# np.random.permutation()这个函数的使用来随机排列一个数组
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# 分割训练集/测试集
"""
dev_sample_index: -1000
"""
# x_train训练集，x_dev验证集，# 直接把dev_sample_index前用于训练,dev_sample_index后的用于训练;
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# print(f'Train/Dev split: {len(y_train)}/{len(y_dev)}')

# 训练
# tensorflow的图结构，#开始构造图
with tf.Graph().as_default():
    # tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
    # 为了避免出现你指定的设备不存在这种情况, 你可以在创建的 session 里把参数 allow_soft_placement 设置为 True,
    # 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        # 定义训练过程
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # 目标函数小于时optimizer,训练结束
        optimizer = tf.train.AdamOptimizer(1e-3)

        # 导入cnn.loss求偏导的差(也就是结果的变化量),反过来就知道,上次的变化,对结果影响
        # 从而知道是否到全局最优解
        grads_and_vars = optimizer.compute_gradients(cnn.loss)

        # 填入训练序列,后面会步进迭代
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # 跟踪梯度值和稀疏性(可选)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(f"{v.name}/grad/hist", g)
                sparsity_summary = tf.summary.scalar(f"{v.name}/grad/sparsity", tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)

        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 模型和摘要的输出目录
        print(f"Writing to {out_dir}\n")

        # 总结损失和准确性
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # 训练总结
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # 开发总结
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # 检查点目录。Tensorflow假设这个目录已经存在，所以我们需要创建它
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 初始化
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            单个训练步骤
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prod: FLAGS.dropout_keep_prod
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(f"{time_str}: step {step}, loss {loss}, acc {accuracy}")
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            在开发集上评估模型
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prod: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        # 生成批次
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # 循环训练。对每一批…
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
