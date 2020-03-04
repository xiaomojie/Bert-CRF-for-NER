#! usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time
import os

import numpy
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers

from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
# from sklearn.metrics import f1_score,precision_score,recall_score
# from tensorflow.python.ops import math_ops
import tf_metrics
import pickle

from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping, mark_mapping
from loader import augment_with_pretrained, prepare_dataset
from loader import load_entropy_dict
import itertools


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第二块GPU（从0开始）

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_crf", True, "Whether to use CRF decoding.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_float("bert_dropout_rate", 0.2,
                   "Proportion of dropout for bert embedding.")

flags.DEFINE_float("bilstm_dropout_rate", 0.2,
                   "Proportion of dropout for bilstm.")
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')


###############
flags.DEFINE_boolean("clean",         False,      "clean train folder")
flags.DEFINE_boolean("train",         False,      "Wither train the model")
flags.DEFINE_boolean("add_mark",      False,      "Wither train the model")
flags.DEFINE_boolean("add_entropy",   False,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",       20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("complete_dis_dict_dim",  5,          "embedding size for dis dict, 0 if not used")
flags.DEFINE_integer("complete_loc_dict_dim",  5,          "embedding size for loc dict, 0 if not used")
flags.DEFINE_integer("complete_res_dict_dim",  5,          "embedding size for res dict, 0 if not used")
flags.DEFINE_integer("partial_dis_dict_dim",  5,          "embedding size for dis dict, 0 if not used")
flags.DEFINE_integer("partial_loc_dict_dim",  5,          "embedding size for loc dict, 0 if not used")
flags.DEFINE_integer("partial_res_dict_dim",  5,          "embedding size for res dict, 0 if not used")
flags.DEFINE_integer("pinyin_dis_dict_dim",  5,          "embedding size for dis dict, 0 if not used")
flags.DEFINE_integer("pinyin_loc_dict_dim",  5,          "embedding size for loc dict, 0 if not used")
flags.DEFINE_integer("pinyin_res_dict_dim",  5,          "embedding size for res dict, 0 if not used")
flags.DEFINE_integer("char_dim",      100,        "Embedding size for characters")
flags.DEFINE_integer("mark_dim",      50,        "Embedding size for chars' mark")
flags.DEFINE_integer("lstm_dim",      100,        "Num of hidden units in LSTM")
flags.DEFINE_integer("lstm_mark_dim", 50,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",     "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",            5,          "Gradient clip")
flags.DEFINE_float("dropout",         0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",      50,         "batch size")
flags.DEFINE_float("lr",              0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",      "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",       True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",         False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",         True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",     50,        "maximum training epochs")
flags.DEFINE_integer("steps_check",   100,        "steps per checkpoint")
flags.DEFINE_string("model_path",      "model",      "Path to save model")
flags.DEFINE_string("summary_path",   "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",       "train.log",    "File for log")
flags.DEFINE_string("map_file",       "maps.pkl",     "file for maps")
# flags.DEFINE_string("vocab_file",     "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",    "config_file",  "File for config")
flags.DEFINE_string("script",         "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",    "result",       "Path for results")
flags.DEFINE_string("cate_file",      os.path.join("loader", "cate"),  "Path for cate")
flags.DEFINE_string("ser_cate_file",  os.path.join("loader", "server_cate"),  "Path for server_cate")
flags.DEFINE_string("after_process_dict",  os.path.join("loader", "after_process_dict"),  "Path for after_process_dict")
flags.DEFINE_string("entropy_dict",   os.path.join("loader", "entropy_dict"),  "Path for entropy_dict")
flags.DEFINE_string("emb_file",       os.path.join("loader", "wiki_100.utf8"), "Path for pre_trained embedding")
# flags.DEFINE_string("train_file",     os.path.join("data_dis", "ner.train"),  "Path for train data")
# flags.DEFINE_string("dev_file",       os.path.join("data", "ner.dev"),    "Path for dev data")
# flags.DEFINE_string("test_file",      os.path.join("data", "ner.test"),   "Path for test data")
# flags.DEFINE_string("test_file",      os.path.join("data", "ner.dev"),   "Path for test data")

FLAGS = tf.app.flags.FLAGS
log_file = os.path.join(FLAGS.model_path, 'log', FLAGS.log_file)
map_file = os.path.join(FLAGS.model_path, FLAGS.map_file)
# vocab_file = os.path.join(FLAGS.model_path, FLAGS.vocab_file)
config_file = os.path.join(FLAGS.model_path, FLAGS.config_file)
result_path = os.path.join(FLAGS.model_path, FLAGS.result_path)
ckpt_path = os.path.join(FLAGS.model_path, 'ckpt')

###########
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                # 一句话结束，则将该句话使用" "分割拼接
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir, data=None):  # 返回的每一个元素是一个InputExample对象
        if not data:
            return self._create_example(
                self._read_data(os.path.join(data_dir, "ner.train")), "train"  # 返回的是[label, word]，label和word分别用" "分割
            )
        else:
            return self._create_example(data, "train")

    def get_dev_examples(self, data_dir, data=None):
        if not data:
            return self._create_example(
                self._read_data(os.path.join(data_dir, "ner.dev")), "dev"
            )
        else:
            return self._create_example(data, "dev")

    def get_test_examples(self, data_dir, data=None):  # 返回的每一个元素是一个InputExample对象
        if not data:
            return self._create_example(
                self._read_data(os.path.join(data_dir, "ner.dev")), "test")
        else:
            return self._create_example(data, "test")

    def get_labels(self):
        # prevent potential bug for chinese text mixed with english text
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
        return ["O", "B-RES", "I-RES", "X", "[CLS]", "[SEP]"]
        # return ["O", "B-DIS", "I-DIS", "B-LOC", "I-LOC", "B-RES", "I-RES", "X", "[CLS]", "[SEP]"]
        # 会在原来标签的基础之上多添加"X","[CLS]", "[SEP]"这三个标签，句子开始设置CLS 标志，句尾添加[SEP] 标志,
        # "X"表示的是英文中缩写拆分时，拆分出的几个部分，除了第1部分，其他的都标记为"X"

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)  # train-0，train-1...
            # text = tokenization.convert_to_unicode(line[1])  # line[1]为"word1 word2 ..."，utf-8转unicode
            tf.logging.info("text: %s" % (line[-1]))
            tf.logging.info("label: %s" % (line[0]))

            text = tokenization.convert_to_unicode(line[-1])  # line[1]为"word1 word2 ..."，utf-8转unicode
            label = tokenization.convert_to_unicode(line[0])  # line[0]为"label1 label2 ..."
            examples.append(InputExample(guid=guid, text=text, label=label))  # exsample中的每一个元素是一个InputExample对象
        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, mode):
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    # print(textlist)
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)  # 使用bert中的tokenizer对输入进行处理
        # print(token)
        tokens.extend(token)
        label_1 = labellist[i]
        # print(label_1)
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
        # print(tokens, labels)
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        # if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    write_tokens(ntokens, mode)
    return feature, ntokens, label_ids


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):  # 第二个参数为下标起始位置
        label_map[label] = i  # label：id
    with open(FLAGS.output_dir + '/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)

    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, ntokens, label_ids = convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer,
                                                             mode)

        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    # sentence token in each batch
    writer.close()
    return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    embedding = model.get_sequence_output()  # (batch_size, seq_length, embedding_size)
    '''
    embedding_1 = model.get_all_encoder_layers()[-2]
    embedding_2 = model.get_all_encoder_layers()[-1]
    embedding = tf.concat([embedding_1, embedding_2], axis=-1)
    '''
    if is_training:
        # dropout embedding
        embedding = tf.layers.dropout(embedding, rate=FLAGS.bert_dropout_rate, training=is_training)
    embedding_size = embedding.shape[-1].value  # embedding_size
    seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # (batch_size)
    tf.logging.info('seq_length: %s', seq_length)
    tf.logging.info('lengths: %s', lengths)

    def bi_lstm_fused(inputs, lengths, rnn_size, is_training, dropout_rate=0.2, scope='bi-lstm-fused'):
        with tf.variable_scope(scope):
            t = tf.transpose(inputs, perm=[1, 0, 2])  # Need time-major
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
            return tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

    def lstm_layer(inputs, lengths, is_training):
        rnn_output = tf.identity(inputs)
        for i in range(2):
            scope = 'bi-lstm-fused-%s' % i
            rnn_output = bi_lstm_fused(rnn_output,
                                       lengths,
                                       rnn_size=FLAGS.lstm_size,
                                       is_training=is_training,
                                       dropout_rate=FLAGS.bilstm_dropout_rate,
                                       scope=scope)  # (batch_size, seq_length, 2*rnn_size)
        return rnn_output

    def project_layer(inputs, out_dim, seq_length, scope='project'):
        with tf.variable_scope(scope):
            in_dim = inputs.get_shape().as_list()[-1]
            weight = tf.get_variable('W', shape=[in_dim, out_dim],
                                     dtype=tf.float32, initializer=initializers.xavier_initializer())
            bias = tf.get_variable('b', shape=[out_dim], dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
            t_output = tf.reshape(inputs, [-1, in_dim])  # (batch_size*seq_length, in_dim)
            output = tf.matmul(t_output, weight) + bias  # (batch_size*seq_length, out_dim)
            output = tf.reshape(output, [-1, seq_length, out_dim])  # (batch_size, seq_length, out_dim)
            return output

    def loss_layer(logits, labels, num_labels, lengths, input_mask):
        trans = tf.get_variable(
            "transitions",
            shape=[num_labels, num_labels],
            initializer=initializers.xavier_initializer())
        if FLAGS.use_crf:
            with tf.variable_scope("crf-loss"):
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=labels,
                    transition_params=trans,
                    sequence_lengths=lengths)
                per_example_loss = -log_likelihood
                loss = tf.reduce_mean(per_example_loss)
                return loss, per_example_loss, trans
        else:
            labels_one_hot = tf.one_hot(labels, num_labels)
            cross_entropy = labels_one_hot * tf.log(tf.nn.softmax(logits))
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
            cross_entropy *= tf.to_float(input_mask)
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
            cross_entropy /= tf.cast(lengths, tf.float32)
            per_example_loss = cross_entropy
            loss = tf.reduce_mean(per_example_loss)
            return loss, per_example_loss, trans

    '''    
    # 1
    logits = project_layer(embedding, num_labels, seq_length, scope='project')
    '''
    '''
    # 2
    lstm_outputs = lstm_layer(embedding, lengths, is_training)
    p1 = project_layer(lstm_outputs, FLAGS.lstm_size, seq_length, scope='project-1')
    p2 = project_layer(p1, num_labels, seq_length, scope='project-2')
    logits = p2
    '''
    # 3
    lstm_outputs = lstm_layer(embedding, lengths, is_training)
    logits = project_layer(lstm_outputs, num_labels, seq_length, scope='project')
    loss, per_example_loss, trans = loss_layer(logits, labels, num_labels, lengths, input_mask)
    if FLAGS.use_crf:
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=lengths)
    else:
        probabilities = tf.nn.softmax(logits, axis=-1)
        pred_ids = tf.argmax(probabilities, axis=-1)

    # masking for confirmation
    pred_ids *= input_mask

    tf.logging.info('#' * 20)
    tf.logging.info('shape of output_layer: %s' %embedding.shape)
    tf.logging.info('embedding_size:%d' % embedding_size)
    tf.logging.info('seq_length:%d' % seq_length)
    tf.logging.info('shape of logit: %s' %logits.shape)
    tf.logging.info('shape of loss:%s' %loss.shape)
    tf.logging.info('shape of per_example_loss:%s'%per_example_loss.shape)
    tf.logging.info('num labels:%d' % num_labels)
    tf.logging.info('#' * 20)
    return (loss, per_example_loss, logits, trans, pred_ids)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        tf.logging.info('shape of input_ids: %s', input_ids.shape)
        tf.logging.info('shape of label_ids: %s', label_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, trans, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tf.logging.info('shape of pred_ids: %s', pred_ids.shape)

        global_step = tf.train.get_or_create_global_step()
        # add summary
        tf.summary.scalar('loss', total_loss)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint and is_training:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=pred_ids,
                scaffold_fn=scaffold_fn
            )
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                '''
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
                '''
                lr = tf.train.exponential_decay(learning_rate, global_step, 5000, 0.9, staircase=True)
                optimizer = tf.train.AdamOptimizer(lr)
                grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 1.5)
                train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
                logging_hook = tf.train.LoggingTensorHook({"batch_loss" : total_loss}, every_n_iter=10)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks = [logging_hook],
                    scaffold_fn=scaffold_fn)
            else: # mode == tf.estimator.ModeKeys.EVAL:
                def metric_fn(label_ids, pred_ids, per_example_loss, input_mask):
                    # ['<pad>'] + ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "X"]
                    indices = [2, 3]
                    precision = tf_metrics.precision(label_ids, pred_ids, num_labels, indices, input_mask)
                    recall = tf_metrics.recall(label_ids, pred_ids, num_labels, indices, input_mask)
                    f = tf_metrics.f1(label_ids, pred_ids, num_labels, indices, input_mask)
                    accuracy = tf.metrics.accuracy(label_ids, pred_ids, input_mask)
                    loss = tf.metrics.mean(per_example_loss)
                    return {
                        'eval_precision': precision,
                        'eval_recall': recall,
                        'eval_f': f,
                        'eval_accuracy': accuracy,
                        'eval_loss': loss,
                    }
                eval_metrics = (metric_fn, [label_ids, pred_ids, per_example_loss, input_mask])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):

    token = batch_tokens[i]
    # if token != "[PAD]" and token != "[CLS]" and true_l != "X":
    if token != "[SEP]" and token != "[CLS]" and token != "**NULL**":
        true_l = id2label[batch_labels[i]]
        if true_l == "X":
            return

        # if prediction == 0:
        #     prediction = 1
        predict = id2label[prediction]

        if predict == "X" and not predict.startswith("##"):
            predict = "O"
        line = "{}\t{}\t{}\n".format(token, true_l, predict)
        wf.write(line)
    if token == "[SEP]":
        wf.write('\n')


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
    with open(output_predict_file, 'w') as wf:

        # if FLAGS.crf:
        #     predictions = []
        #     for m, pred in enumerate(result):
        #         predictions.extend(pred)
        #     for i, prediction in enumerate(predictions):
        #         _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)
        #
        # else:

        for i, prediction in enumerate(result):
            # for prediction_id in prediction:
            #     _write_base(batch_tokens, id2label, prediction_id, batch_labels, wf, i)
            _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)
        # with open(output_predict_file,'w') as writer:
        #     for prediction in result:
        #         output_line = "\n".join(id2label[id] for id in prediction if id!=0) + "\n"  # 这里的prediction是一个数组，以句子为单位
        #         writer.write(output_line)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "ner": NerProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)  # 加载bert模型的参数设置

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:  # 限制ner的max_seq_length不大于bert的最大长度限制512
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()  # 获取label标签["O", "B-DIS", "I-DIS", "X", "[CLS]", "[SEP]"]

    tokenizer = tokenization.FullTokenizer(  # 对vocab的初始处理，包括word:id，大小写等
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:  # use_tpu 默认为False
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,  # how often to save the model checkpoint. 1000
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,  # 1000
            num_shards=FLAGS.num_tpu_cores,  # 8
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None  # warm up 步数的比例，比如说总共学习100步，warmup_proportion=0.1表示前10步用来warm up，warm up时以
    # 较低的学习率进行学习(lr = global_step/num_warmup_steps * init_lr)，10步之后以正常(或衰减)的学习
    # 率来学习。

    ##################
    train_sentences = load_sentences(os.path.join(FLAGS.data_dir, "ner.train"), FLAGS.lower,
                                     FLAGS.zeros)  # 加载训练数据，格式为二维list，外层存储每一句话，内层为每句话的一个字和对应的tag
    dev_sentences = load_sentences(os.path.join(FLAGS.data_dir, "ner.dev"), FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(os.path.join(FLAGS.data_dir, "ner.dev"), FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)  # 默认IOBES，更新tag方案，将IOB转化为IOBES
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)  # 默认IOBES，更新tag方案，将IOB转化为IOBES
    update_tag_scheme(test_sentences, FLAGS.tag_schema)


    # create maps if not exist
    if not os.path.isfile(map_file):
        # create dictionary for word
        if FLAGS.pre_emb:    # use pre-trained embedding
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(  # 为了保证训练集中未出现的测试集中的字至少也能用预训练的word embedding
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])  # 将嵌套的列表拼接
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        # 执行mark_mapping
        _c, mark_to_id, id_to_mark = mark_mapping(train_sentences)

        entropy_dict = load_entropy_dict(FLAGS.entropy_dict)

        with open(map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag, mark_to_id, id_to_mark, entropy_dict], f)
    else:
        with open(map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag, mark_to_id, id_to_mark, entropy_dict = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, mark_to_id, entropy_dict, FLAGS.lower
    )
    dev_data = prepare_dataset(
         dev_sentences, char_to_id, tag_to_id, mark_to_id, entropy_dict, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, mark_to_id, entropy_dict, FLAGS.lower
    )


    ###############




    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir, train_data)  # 返回的每一个元素是一个InputExample对象
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,  # 将预训练的bert模型的参数加载到模型中作为fine-tuning的初始化参数
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    filed_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    filed_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
    with open(FLAGS.output_dir + '/label2id.pkl', 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    if os.path.exists(token_path):
        os.remove(token_path)
    predict_examples = processor.get_test_examples(FLAGS.data_dir)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    # batch_labels 是以句为单位的[[1,2,0,0,1,2],[...]]
    batch_tokens, batch_labels = filed_based_convert_examples_to_features(predict_examples, label_list,
                                                                          FLAGS.max_seq_length, tokenizer,
                                                                          predict_file, mode="test")

    for actual_train_step in list(range(1000, num_train_steps, 2000)) + [num_train_steps]:

        if FLAGS.do_train:
            start = time.clock()
            tf.logging.info("start training time: %f", start)
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(train_examples))
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Num steps = %d", actual_train_step)
            train_input_fn = file_based_input_fn_builder(
                input_file=train_file,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
            estimator.train(input_fn=train_input_fn, max_steps=actual_train_step)

            end = time.clock()
            tf.logging.info("end training time: %f", end)
            tf.logging.info("training time: %f", end - start)

        if FLAGS.do_eval:
            start = time.clock()
            tf.logging.info("start evaluation time: %f", start)

            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Num examples = %d", len(eval_examples))
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
            eval_steps = None
            if FLAGS.use_tpu:
                eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
            eval_drop_remainder = True if FLAGS.use_tpu else False
            eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=eval_drop_remainder)
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            end = time.clock()
            tf.logging.info("end evaluation time: %f", end)
            tf.logging.info("evaluation time: %f", end - start)

        if FLAGS.do_predict:
            start = time.clock()
            tf.logging.info("start predict time: %f", start)
            tf.logging.info("***** Running prediction *****")
            tf.logging.info("  Num examples = %d", len(predict_examples))
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
            if FLAGS.use_tpu:
                # Warning: According to tpu_estimator.py Prediction on TPU is an
                # experimental feature and hence not supported here
                raise ValueError("Prediction in TPU not supported")
            predict_drop_remainder = True if FLAGS.use_tpu else False
            predict_input_fn = file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder)

            result = estimator.predict(input_fn=predict_input_fn)

            _result = []
            for prediction in result:
                _result += [prediction_id for prediction_id in prediction]

            output_predict_file = os.path.join(FLAGS.output_dir + "/label_test/", "label_test.txt-"+str(actual_train_step))
            Writer(output_predict_file, _result, batch_tokens, batch_labels, id2label)

            end = time.clock()
            tf.logging.info("end predict time: %f", end)
            tf.logging.info("predict time: %f", end - start)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
