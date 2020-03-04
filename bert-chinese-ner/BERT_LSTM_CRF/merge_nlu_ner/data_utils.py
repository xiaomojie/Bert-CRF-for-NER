# encoding=utf8
import re
import math
import codecs
import random

import numpy as np
import jieba
import pdb
jieba.initialize()


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def create_input(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights


def full_to_half(s):
    """
    Convert full-width character to half-width one 
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def cut_to_sentence(text):
    """
    Cut text to sentences 
    """
    sentence = []
    sentences = []
    len_p = len(text)
    pre_cut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if pre_cut:
            cut=True
            pre_cut=False
        if word in u"。;!?\n":
            cut = True
            if len_p > idx+1:
                if text[idx+1] in ".。”\"\'“”‘’?!":
                    cut = False
                    pre_cut=True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def input_from_line(sentence, char_to_id, mark_to_id, entropy_dict):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = []
    mark = []
    dis_complete_tag = []
    dis_partial_tag = []
    dis_pinyin_tag = []
    loc_complete_tag = []
    loc_partial_tag = []
    loc_pinyin_tag = []
    res_complete_tag = []
    res_partial_tag = []
    res_pinyin_tag = []
    dis_dict_map = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
    loc_dict_map = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
    res_dict_map = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
    for elem in sentence:
        line.append(elem[0])
        mark.append(elem[1])
        dis_complete_tag.append(dis_dict_map[elem[2]])
        dis_partial_tag.append(dis_dict_map[elem[3]])
        dis_pinyin_tag.append(dis_dict_map[elem[4]])
        loc_complete_tag.append(loc_dict_map[elem[5]])
        loc_partial_tag.append(loc_dict_map[elem[6]])
        loc_pinyin_tag.append(loc_dict_map[elem[7]])
        res_complete_tag.append(res_dict_map[elem[8]])
        res_partial_tag.append(res_dict_map[elem[9]])
        res_pinyin_tag.append(res_dict_map[elem[10]])
    line = "".join(line)
    # line = full_to_half(line)
    # line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[mark_to_id[w if w in mark_to_id else '<UNK>'] for w in mark]])
    inputs.append([dis_complete_tag])
    inputs.append([dis_partial_tag])
    inputs.append([dis_pinyin_tag])
    inputs.append([loc_complete_tag])
    inputs.append([loc_partial_tag])
    inputs.append([loc_pinyin_tag])
    inputs.append([res_complete_tag])
    inputs.append([res_partial_tag])
    inputs.append([res_pinyin_tag])
    # print("charInputs: ", inputs[1])
    # print("segInputs: ", inputs[2])
    # print("dis_complete_tag: ", dis_complete_tag)
    # print("dis_partial_tag: ", dis_partial_tag)
    # print("dis_pinyin_tag: ", dis_pinyin_tag)
    # print("res_complete_tag: ", res_complete_tag)
    # print("res_partial_tag: ", res_partial_tag)
    # print("res_pinyin_tag: ", res_pinyin_tag)
    # print("loc_complete_tag: ", loc_complete_tag)
    # print("loc_partial_tag: ", loc_partial_tag)
    # print("loc_pinyin_tag: ", loc_pinyin_tag)
    left_entropy = [entropy_dict[w][0] if w in entropy_dict else 0 for w in line]
    left_entropy_average = np.mean(np.array(left_entropy))
    left_entropy = [elem if elem != 0 else left_entropy_average for elem in left_entropy]
    if max(left_entropy) != min(left_entropy):
        left_entropy = [[(i - min(left_entropy)) / (max(left_entropy) - min(left_entropy))] for i in left_entropy]
    else:
        left_entropy = [[i] for i in left_entropy]
    inputs.append([left_entropy])

    right_entropy = [entropy_dict[w][1] if w in entropy_dict else 0 for w in line]
    right_entropy_average = np.mean(np.array(right_entropy))
    right_entropy = [elem if elem != 0 else right_entropy_average for elem in right_entropy]
    if max(right_entropy) != min(right_entropy):
        right_entropy = [[(i - min(right_entropy)) / (max(right_entropy) - min(right_entropy))] for i in right_entropy]
    else:
        right_entropy = [[i] for i in right_entropy]
    inputs.append([right_entropy])

    inputs.append([[]])
    return inputs


class BatchManager(object):

    def __init__(self, data,  batch_size, sort=True):
        self.batch_data = self.sort_and_pad(data, batch_size, sort)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size, sort=True):
        batch_size = int(batch_size)
        num_batch = int(math.ceil(len(data) / batch_size))
        if sort:
            sorted_data = sorted(data, key=lambda x: len(x[0]))
        else:
            sorted_data = data
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size: (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        marks = []
        complete_dis_dicts = []
        partial_dis_dicts = []
        pinyin_dis_dicts = []
        complete_loc_dicts = []
        partial_loc_dicts = []
        pinyin_loc_dicts = []
        complete_res_dicts = []
        partial_res_dicts = []
        pinyin_res_dicts = []
        left_entropy = []
        right_entropy = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, mark, complete_dis_dict, partial_dis_dict, pinyin_dis_dict,\
                complete_loc_dict, partial_loc_dict, pinyin_loc_dict, \
                complete_res_dict, partial_res_dict, pinyin_res_dict, left_ent, right_ent, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            marks.append(mark + padding)
            complete_dis_dicts.append(complete_dis_dict + padding)
            partial_dis_dicts.append(partial_dis_dict + padding)
            pinyin_dis_dicts.append(pinyin_dis_dict + padding)
            complete_loc_dicts.append(complete_loc_dict + padding)
            partial_loc_dicts.append(partial_loc_dict + padding)
            pinyin_loc_dicts.append(pinyin_loc_dict + padding)
            complete_res_dicts.append(complete_res_dict + padding)
            partial_res_dicts.append(partial_res_dict + padding)
            pinyin_res_dicts.append(pinyin_res_dict + padding)
            padding_ent = [[0]] * (max_length - len(string))
            left_entropy.append(left_ent + padding_ent)
            right_entropy.append(right_ent + padding_ent)
            targets.append(target + padding)
        return [strings, chars, segs, marks, complete_dis_dicts, partial_dis_dicts, pinyin_dis_dicts,
                complete_loc_dicts, partial_loc_dicts, pinyin_loc_dicts,
                complete_res_dicts, partial_res_dicts, pinyin_res_dicts, left_entropy, right_entropy, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
