# Author:BinHaoWang
# System Date:2019/1/3
# System Time:18:42   

import tensorflow as tf
import numpy as np
import os
import data_helpers
import word2vec_helpers

import csv

# 参数

# 数据参数
tf.flags.DEFINE_string("input_text_file", "./data/spam_100.utf8", "测试文本数据源")
tf.flags.DEFINE_string("input_label_file", "", "测试文本数据源的标签文件")

# 评估参数
tf.flags.DEFINE_integer("batch_size", 64, "批处理大小(默认:64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "检查点目录从训练运行")
tf.flags.DEFINE_boolean("eval_train", True, "评估所有训练数据")

# Misc 参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "允许设备软设备放置")
tf.flags.DEFINE_boolean("log_device_placement", False, "操作系统在设备上的日志位置")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 验证

# 验证签出点文件
# 查找最新保存的检查点文件的文件名
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print(f"Using checkpoint file : {checkpoint_file}")

# 验证word2vec模型文件
trained_word2vec_model_file = os.path.join(FLAGS.checkpoint_dir, "..", "trained_word2vec.model")
if not os.path.exists(trained_word2vec_model_file):
    print(f"Word2vec model file \'{trained_word2vec_model_file}\' doesn't exist!")
print(f"Using word2vec model file : {trained_word2vec_model_file}")

# 验证培训参数文件
training_params_file = os.path.join(FLAGS.checkpoint_dir, "..", "training_params.pickle")
if not os.path.exists(training_params_file):
    print(f"Training params file \'{training_params_file}\' is missing!")
print(f"Using training params file : {training_params_file}")

# 加载自己保存的参数
params = data_helpers.loadDict(training_params_file)
num_labels = int(params['num_labels'])
max_document_length = int(params['max_document_length'])

# Load data
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.input_text_file, FLAGS.input_label_file, num_labels)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# 获取嵌入向量x_test
sentences, max_document_length = data_helpers.padding_sentences(x_raw, '<PADDING>',
                                                                padding_sentence_length=max_document_length)
x_test = np.array(word2vec_helpers.embedding_sentences(sentences, file_to_load=trained_word2vec_model_file))
print(f"x_test.shape = {x_test.shape}")

# 评估
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 加载保存的元图并恢复变量
        saver = tf.train.import_meta_graph(f"{checkpoint_file}.meta")
        saver.restore(sess, checkpoint_file)

        # 通过名称从图中获取占位符
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prod").outputs[0]

        # 我们要计算张量
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # 为一个纪元生成批处理
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # 在这里收集预测
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# 定义y_test时打印精度
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print(f"Total number of test examples: {len(y_test)}")
    print(f"Accuracy: {correct_predictions / float(len(y_test))}")

# 将计算保存到csv中
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print(f"Saving evaluation to {out_path}")
with open(out_path, 'w', encoding='utf-8') as f:
    csv.writer(f).writerows(predictions_human_readable)
