#!/usr/bin/python
# -*- coding: utf-8 -*-

import codecs
import re
import sys
import copy
import json
import os

reload(sys)
sys.setdefaultencoding('utf-8') 


def read_txt(file_name):
    with codecs.open(file_name, "r") as fr:
        data = fr.read()
    data = [line for line in data.split("\n") if line]
    return data


def write_txt(data, file_name):
    fw = open(file_name, "w+")
    for line in data:
        fw.write(line + "\n")
    fw.close()


def read_json(file_name):
    with open(file_name, 'r') as fr:
        json_data = json.load(fr)
    return json_data


def in_slots(slots, tmp_slot):
    for slot in slots:
        if cmp(slot['text'], tmp_slot['text']) == 0\
            and cmp(slot['slot'], tmp_slot['slot']) == 0 \
                and slot['startIndex'] == tmp_slot['startIndex']:
            return True
    return False


def included_by_other_slot(slots, tmp_slot):
    tmp_slot_start = tmp_slot['startIndex'] - 1
    tmp_slot_end = tmp_slot_start + len(tmp_slot['text']) - 1
    for slot in slots:
        if 'startIndex' in slot:
            slot_start = slot['startIndex'] - 1
            slot_end = slot_start + len(slot['text']) - 1
            if tmp_slot_start >= slot_start and tmp_slot_end < slot_end or \
                    tmp_slot_start > slot_start and tmp_slot_end <= slot_end:
                return True
    return False


slot_name_map = read_json("./data/dict/slot_name_map.json")
dish_cate_list = read_txt("./data/dict/server_cate_cai")
dish_name_list = read_txt("./data/dict/daimler_dish.list")
dish_name_map = {}
for each_dish_name in dish_name_list:
    dish_name_map[each_dish_name] = "1"


def get_occur_pos_list(raw_query, slot):
    pos_list = []
    slot_text = slot['text']
    search_begin = 0
    while search_begin < len(raw_query):
        start_pos = raw_query.find(slot_text, search_begin)
        if start_pos == -1:
            break
        if start_pos != slot['startIndex']-1:
            end_pos = start_pos + len(slot_text) - 1
            pos_list.append((start_pos, end_pos))
        search_begin = start_pos + len(slot_text) 
    return pos_list


def is_overlap_with_other_slots(start_pos, end_pos, raw_slots):
    for slot in raw_slots:
        slot_start = slot['startIndex'] - 1
        slot_end = slot_start + len(slot['text']) - 1
        if start_pos >= slot_start and start_pos <= slot_end or \
            slot_start >= start_pos and slot_start <= slot_end:
            return True
    return False


def clean_slot_by_priority(raw_slots, low_priority_slot="菜名", high_priority_slot="饭店名"):
    result_after_clean = []
    if len(raw_slots) == 1:
        result_after_clean = raw_slots
    else:
        for index_i in range(len(raw_slots)):
            flag = True
            for index_j in range(len(raw_slots)):
                if index_j != index_i:
                    if raw_slots[index_i]["text"] == raw_slots[index_j]["text"] and \
                            raw_slots[index_i]["startIndex"] == raw_slots[index_j]["startIndex"]:
                        if raw_slots[index_i]["slot"] == low_priority_slot and raw_slots[index_j]["slot"] == high_priority_slot:
                            flag = False
                            break
            if flag:
                result_after_clean.append(raw_slots[index_i])

    return result_after_clean


def clean_slots(raw_query, raw_slots):
    new_raw_slots = []
    for slot in raw_slots:
        # 如果跟其他槽位相同，则去掉
        if in_slots(new_raw_slots, slot):
            continue

        if included_by_other_slot(raw_slots, slot):
            continue
        
        new_raw_slots.append(copy.deepcopy(slot))

        # 如果该槽的槽值在句式中其他的部分出现，且未和其他已有的槽位有overlap，那我们把他认为是标注人员没有的标注的槽
        other_pos_list = get_occur_pos_list(raw_query, slot)
        for start_pos, end_pos in other_pos_list:
            if is_overlap_with_other_slots(start_pos, end_pos, raw_slots):
                continue
            else:
                new_slot = copy.deepcopy(slot)
                new_slot['startIndex'] = start_pos + 1
                new_raw_slots.append(new_slot)

    return new_raw_slots


def uniform_slot(raw_query, raw_slots, slot):
    # 根据配置文件，将相应的中文槽位名称换成英文槽位名称
    slot['slot'] = slot_name_map[slot['slot']] 
    slot_name = slot['slot']
    slot_text = slot['text']
    slot_start = slot['startIndex']

    # 原来标注为品类的词，如果该词后有”店“、”馆“、“屋”等表示店的词，连同后面的词一起标注为餐厅名
    slot_end = slot_start + len(slot_text) - 1
    # print(len(raw_query))
    # print(slot_end)
    # print raw_query
    if slot_end < len(raw_query):
        next_word = raw_query[slot_end]
        if next_word == '店' or next_word == '馆' or next_word == '铺':
            slot_text = slot_text + next_word
            slot_name = 'restaurant_name'
            slot['slot'] = slot_name
            slot['text'] = slot_text
        
    # 菜品清洗: 只将菜系和早晚餐、小吃、早点等认为是品类，其他认为是菜名
    if slot_name == "dish_cate":
        if slot_text not in dish_cate_list:
            slot['slot'] = "dish_name"
    if slot_name == "dish_name":
        if slot_text in dish_cate_list:
            slot['slot'] = "dish_cate"
    
    # 商品名如果是在dish列表中，也认为是dish
    if slot_name == "commodity_name":
        if slot_text in dish_name_map:
            slot_new = copy.deepcopy(slot)
            slot_new['slot'] = u'菜名'
            if not in_slots(raw_slots, slot_new):
                raw_slots.append(slot_new)


def tagged_data_deal(tagged_file_name, entity_flag):
    print("entity_flag: ", entity_flag)
    print("entity_flag_dis: ", entity_flag["dis_flag"])
    print("entity_flag_res: ", entity_flag["res_flag"])
    print("entity_flag_loc: ", entity_flag["loc_flag"])

    data = read_txt(tagged_file_name)
    all_query = []
    all_raw_query = []
    for line in data:
        deal_tagging_result = {}
        query_tagging_result = json.loads(line)
        if 'content'not in query_tagging_result:
            continue
        raw_query = json.loads(query_tagging_result['content'])['text']
        
        all_raw_query.append(raw_query)
        data = query_tagging_result['data']
        if 'label1' in data:
            tagging_result = json.loads(query_tagging_result['data']['label1'])
        else:
            tagging_result = json.loads(query_tagging_result['data']['label0'])
            
        raw_slots = tagging_result["slots"] if "slots" in tagging_result else []

        raw_slots = clean_slots(raw_query, raw_slots)

        # 相同str同时填入菜名和饭店名槽位，则删掉菜名槽
        raw_slots = clean_slot_by_priority(raw_slots)
        raw_slots = clean_slot_by_priority(raw_slots, "饭店名", "地址")
        raw_slots = clean_slot_by_priority(raw_slots, "地址", "菜名")

        slots = []
        for index, each_slot in enumerate(raw_slots):
           
            slot_text = raw_slots[index]['text']
            uniform_slot(raw_query, raw_slots, each_slot)

            slot_tmp = {}
            if raw_slots[index]['slot'] == 'location' and entity_flag["loc_flag"] or\
               raw_slots[index]['slot'] == 'dish_name' and entity_flag["dis_flag"] or\
               raw_slots[index]['slot'] == 'restaurant_name' and entity_flag["res_flag"]:
                slot_tmp["text"] = raw_slots[index]['text']
                slot_tmp["slot"] = raw_slots[index]['slot']
                slot_tmp['startIndex'] = raw_slots[index]['startIndex']
                # print(slot_tmp["text"])
                if not in_slots(slots, slot_tmp):
                    slots.append(copy.deepcopy(slot_tmp))
        deal_tagging_result['query'] = raw_query
        deal_tagging_result['slot'] = slots

        if len(deal_tagging_result['slot']) != 0:
            # 初始化label_list
            label_list = ['-']*(len(raw_query)+1)

            for elem_map in deal_tagging_result['slot']:
                slot_text = elem_map["text"]
                slot_name = elem_map["slot"]
                slot_start = elem_map["startIndex"] - 1
                slot_end = slot_start + len(slot_text)
                pos = raw_query.find(elem_map["text"])
                if pos != -1:
                    if elem_map["slot"] == "location":
                        if label_list[slot_start] == '-':
                            label_list[slot_start] = '['
                        else:
                            label_list[slot_start] = label_list[slot_start] + '['
                        if label_list[slot_end] == '-':
                            label_list[slot_end] = ']LOC'
                        else:
                            label_list[slot_end] = ']LOC' + label_list[slot_end]
                        # raw_query = re.sub(elem_map["text"], "[" + elem_map["text"] + "]LOC", raw_query)
                    elif elem_map["slot"] == "dish_name":
                        # raw_query = re.sub(elem_map["text"], "[" + elem_map["text"] + "]DIS", raw_query)
                        if label_list[slot_start] == '-':
                            label_list[slot_start] = '['
                        else:
                            label_list[slot_start] = label_list[slot_start] + '['
                        if label_list[slot_end] == '-':
                            label_list[slot_end] = ']DIS'
                        else:
                            label_list[slot_end] = ']DIS' + label_list[slot_end]
                    elif elem_map["slot"] == "restaurant_name":
                        # raw_query = re.sub(elem_map["text"], "[" + elem_map["text"] + "]RES", raw_query)
                        if label_list[slot_start] == '-':
                            label_list[slot_start] = '['
                        else:
                            label_list[slot_start] = label_list[slot_start] + '['
                        if label_list[slot_end] == '-':
                            label_list[slot_end] = ']RES'
                        else:
                            label_list[slot_end] = ']RES' + label_list[slot_end]
            
            label_query = ''
            for index, token in enumerate(list(raw_query)):
                if label_list[index] != '-':
                    label_query = label_query + label_list[index]
                label_query = label_query + token
            if label_list[-1] != "-":
                label_query = label_query + label_list[-1]
            
        else:
            label_query = raw_query
        # print(raw_query + "-->" + label_query)
        all_query.append(label_query)
        
    return all_query, all_raw_query


def main(file_in, file_out, entity_flag):
    all_query = []
    for each_file_name in file_in:
        query, raw_query = tagged_data_deal(each_file_name, entity_flag)
        all_query += query
    write_txt(all_query, file_out)


if __name__ == "__main__":
    dir_path_in = sys.argv[1]
    file_name_out = sys.argv[2]
    dis_flag = sys.argv[3]
    res_flag = sys.argv[4]
    loc_flag = sys.argv[5]
    file_name = os.listdir(dir_path_in)
    if ".DS_Store" in file_name:
        file_name.remove(".DS_Store")
    file_name_in = [os.path.join(dir_path_in, elem) for elem in file_name]
    dis_flag = True if dis_flag == "true" else False
    res_flag = True if res_flag == "true" else False
    loc_flag = True if loc_flag == "true" else False
    entity_flag = {"dis_flag": dis_flag, "res_flag": res_flag, "loc_flag": loc_flag}
    main(file_name_in, file_name_out, entity_flag)
