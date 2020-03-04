#!/system/bin/sh

#================================================================
#   Copyright (C) 2019 Meituan Inc. All rights reserved.
#   
#   File Name：trans.sh
#   Author：Your Name
#   Created Time：2019-03-28
#   Description：
#
#==============================================================

input_data=$1
if [ $# != 1 ];then
    echo "please input data path: train_data | dev_data | test_data"
    exit 1
fi

#需手动配置的参数
include_dis=true
include_res=true
include_loc=true

input_path_list=("./data/train_data" "./data/dev_data" "./data/test_data")
output_file_list=("ner.train" "ner.dev" "ner.test")
if [[ $input_data == "train_data" ]];then
    input_path=${input_path_list[0]}
    output_file=${output_file_list[0]}
elif [[ $input_data == "dev_data" ]];then
    input_path=${input_path_list[1]}
    output_file=${output_file_list[1]}
else
    input_path=${input_path_list[2]}
    output_file=${output_file_list[2]}
fi
echo "input_data_path: $input_path"
echo "output_file_name: $output_file"

python tagged_data_to_query.py $input_path query.txt $include_dis $include_res $include_loc

python query_to_train_data.py query.txt query_standard_format.txt

if [[ $input_data == "train_data" ]];then
    if [[ $include_dis == true ]];then
        cat ./data/taocan_data/taocan.small.data_dis >> query_standard_format.txt
        echo "use taocan.small.data_dis"
    else
        cat ./data/taocan_data/taocan.small.data_nodis >> query_standard_format.txt
        echo "use taocan.small.data_nodis"
    fi
fi

if [[ $input_data == "train_data" ]];then
    mv query.txt query_train.txt
    mv query_standard_format.txt ner.train
elif [[ $input_data == "dev_data" ]];then
    mv query.txt query_dev.txt
    mv query_standard_format.txt ner.dev
else
    mv query.txt query_test.txt
    mv query_standard_format.txt ner.test
fi