import os
dir = "./output_dis/result_dir/label_test/"
names = os.listdir(dir)
names.sort()
for name in names:
    name = dir + name
    os.system("echo " + name + ">> " + dir + "entity_predict_result.txt")
    os.system("python conlleval.py " + name + " >> " + dir + "entity_predict_result.txt")

