#!/usr/bin/python
# -*- coding: utf-8 -*- 

import codecs
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')  # 设置默认编码格式为'utf-8'


def read_data(file_name):
    with codecs.open(file_name, 'r', encoding="utf8") as fr:
        data = fr.read()
    data = [i for i in data.split("\n") if i]
    return data


def tagging_sentences(sentences):
    train_data = []
    train_data_tmp = []
    entity = {"LOC": [], "RES": [], "DIS": [], "SAT": [], "CAT": []}
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            if sentence[i] == "[":
                j = i
                j += 1
                while sentence[j] != "]":
                    j += 1
                entity[sentence[(j + 1):(j + 4)]].append(sentence[(i + 1):j])
                train_data_tmp.append(sentence[i + 1])
                tag = "B-" + sentence[(j + 1):(j + 4)]
                train_data_tmp.append(tag)
                i += 1
                while j - i - 1 > 0:
                    train_data_tmp.append(sentence[i + 1])
                    tag = "I-" + sentence[(j + 1):(j + 4)]
                    train_data_tmp.append(tag)
                    i += 1
                i += 5
            else:
                train_data_tmp.append(sentence[i])
                tag = "O"
                train_data_tmp.append(tag)
                i += 1
        train_data.append(train_data_tmp)
        train_data_tmp = []
    return train_data, entity


def write_txt(data, file_name):
    fw = open(file_name, "w+")
    for sentence in data:
        for index in range(int(len(sentence) / 2)):
            fw.write(sentence[2 * index] + " " + sentence[2 * index + 1] + "\n")
        fw.write("\n")
    fw.close()


def query_to_train_data(file_in, file_out):
    test_data = read_data(file_in)
    example_test, loc_res_test = tagging_sentences(test_data)
    write_txt(example_test, file_out)


def main(file_in, file_out):
    query_to_train_data(file_in, file_out)


if __name__ == '__main__':
    file_name_in = sys.argv[1]
    file_name_out = sys.argv[2]
    main(file_name_in, file_name_out)



















