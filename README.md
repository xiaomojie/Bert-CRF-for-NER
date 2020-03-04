美团命名实体识别项目
目标：识别对话数据中的菜名(dis)、商家名(res)和地名(loc)

1. 对google bert使用美团评论数据进行进行预训练，得到更加适合美团领域的bert模型
2. BERT_NER.py: bert+softmax进行实体识别
3. BERT_LSTM_CRF：在bert后面接上lstm和crf模型进行命名实体识别
4. train_data_generate: 自动方式生成训练数据，另一部分训练数据为人工标注数据