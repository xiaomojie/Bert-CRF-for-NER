# -*- coding: utf-8 -*-
import os
import re
import codecs
import math

import numpy as np
from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r'):
        num += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word = line.split()
            assert len(word) >= 2, str([word[0]])
            #assert len(word) >= 2, "test" 
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]  # 取出每个字的标签
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)  # 构造一个输入token的词典
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)  # 创建两个映射表
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def mark_mapping(sentences):
    """
    Create a dictionary and a mapping of marks, sorted by frequency.
    """
    marks = [[mark[1] for mark in s] for s in sentences]
    dico = create_dico(marks)
    dico['<UNK>'] = 10000000
    mark_to_id, id_to_mark = create_mapping(dico)
    return dico, mark_to_id, id_to_mark


def load_entropy_dict(file_name_in):
    entropy_dict = {}
    with codecs.open(file_name_in, "r") as fr:
        data = fr.read()
    data = [elem for elem in data.split("\n") if elem]
    for line in data:
        line = line.split("\t")
        entropy_dict[line[0]] = [float(line[1]), float(line[2])]
    return entropy_dict


def prepare_dataset(sentences, char_to_id, tag_to_id, mark_to_id, entropy_dict, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]  # 目前train始终未true，none_index并未作用

    def f(x):
        return x.lower() if lower else x
    data = []
    dis_dict_map = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
    loc_dict_map = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
    res_dict_map = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
    for s in sentences:
        string = ' '.join([w[0] for w in s])
        chars = ' '.join([char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string])
        segs = ' '.join(get_seg_features("".join(string)))
        marks = ' '.join([mark_to_id[w[1] if w[1] in mark_to_id else '<UNK>'] for w in s])
        complete_dis_dicts = ' '.join([dis_dict_map[w[1]] for w in s])
        partial_dis_dicts = ' '.join([dis_dict_map[w[2]] for w in s])
        pinyin_dis_dicts = ' '.join([dis_dict_map[w[3]] for w in s])
        complete_loc_dicts = ' '.join([loc_dict_map[w[4]] for w in s])
        partial_loc_dicts = ' '.join([loc_dict_map[w[5]] for w in s])
        pinyin_loc_dicts = ' '.join([loc_dict_map[w[6]] for w in s])
        complete_res_dicts = ' '.join([res_dict_map[w[7]] for w in s])
        partial_res_dicts = ' '.join([res_dict_map[w[8]] for w in s])
        pinyin_res_dicts = ' '.join([res_dict_map[w[9]] for w in s])
        # 左右熵计算(不存在时存储多少有待考虑，目前用均值填充)
        left_entropy = [entropy_dict[w][0] if w in entropy_dict else 0 for w in string]
        # 计算非0的均值可能会出bug：一个query仅有一个字非0，其他均为0时会出现除0；这里用整体均值
        # left_entropy_average = np.mean(np.array(left_entropy)[np.array(left_entropy) != 0])
        left_entropy_average = np.mean(np.array(left_entropy))
        left_entropy = [elem if elem != 0 else left_entropy_average for elem in left_entropy]
        if max(left_entropy) != min(left_entropy):
            left_entropy = [[(i - min(left_entropy)) / (max(left_entropy) - min(left_entropy))] for i in left_entropy]
        else:
            left_entropy = [[i] for i in left_entropy]

        left_entropy = ' '.join(left_entropy)

        right_entropy = [entropy_dict[w][1] if w in entropy_dict else 0 for w in string]
        # right_entropy_average = np.mean(np.array(right_entropy)[np.array(right_entropy) != 0])
        right_entropy_average = np.mean(np.array(right_entropy))
        right_entropy = [elem if elem != 0 else right_entropy_average for elem in right_entropy]
        if max(right_entropy) != min(right_entropy):
            right_entropy = [[(i - min(right_entropy)) / (max(right_entropy) - min(right_entropy))] for i in right_entropy]
        else:
            right_entropy = [[i] for i in right_entropy]
        right_entropy = ' '.join(right_entropy)

        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        tags = ' '.join(tags)

        data.append([string, chars, segs, marks, complete_dis_dicts, partial_dis_dicts, pinyin_dis_dicts,
                     complete_loc_dicts, partial_loc_dicts, pinyin_loc_dicts,
                     complete_res_dicts, partial_res_dicts, pinyin_res_dicts, left_entropy, right_entropy, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

