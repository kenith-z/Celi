import argparse

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import time
from torch.utils.tensorboard import SummaryWriter

from bidirectional_dict import BidirectionalDict

from bid_lstm_model import BiLstmModel
from word_dataset import WordDataset

from util import decode_word
import pickle
import sys
import torch.onnx
from util import encode_word


def train(dataset, model, args):
    """训练"""
    writer = SummaryWriter(args.logdir)

    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    # 学习率
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.max_epochs):

        state_h, state_c = model.init_state(args.num_layers, args.sequence_length, args.hidden_size, device)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.to(device)  # 提取当前句子的张量表示
            y = y.to(device)  # 提取下一个句子的张量表示

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))

            loss = criterion(y_pred.transpose(1, 2), y).to(device)
            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            if batch % args.save_interval == 0:
                torch.save(model, model_save_path)

            # 使用SummaryWriter记录训练的损失值
            writer.add_scalar('损失值', loss.item(), epoch * len(dataloader) + batch)
    writer.close()  # 训练完成后关闭SummaryWriter


def predict(bd, model, text, next_words=20):
    """验证"""
    words = list(text)
    model.eval()
    state_h, state_c = model.init_state(args.num_layers, len(words), args.hidden_size, device)

    for i in range(0, next_words):
        input_list = [encode_word(bd, w) for w in words[i:]]
        x = torch.tensor([[sublist[0] for sublist in input_list]]).to(device)
        # if i==0:
        #     torch.onnx.export(model, (x, (state_h.to(device), state_c.to(device))), './model.onnx')

        y_pred, (state_h, state_c) = model(x, (state_h.to(device), state_c.to(device)))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()

        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(decode_word(bd, word_index))

    return "".join(words)


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    print("训练设备:{}".format(device))
    parser = argparse.ArgumentParser(description='lstm')
    parser.add_argument('--train-novel-path', type=str, default='datasets')
    parser.add_argument('--max-epochs', type=int, default=1)  # 训练多少遍 总的文本  , default=20)
    parser.add_argument('--batch-size', type=int, default=64)  # default=256)
    parser.add_argument('--sequence-length', type=int, default=75)  # sequence-length 每次训练多长的句子, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.01)  # 添加学习率参数
    parser.add_argument('--save-interval', type=int, default=500)  # 添加模型保存间隔参数
    parser.add_argument('--logdir', type=str, default='./tf-logs')  # 添加模型保存间隔参数

    input_size = 256
    parser.add_argument('--input_size', type=int, default=input_size)  # 输入大小
    parser.add_argument('--hidden_size', type=int, default=256)  # 隐藏层大小
    parser.add_argument('--embedding_dim', type=int, default=input_size)  # 单词向量
    parser.add_argument('--num_layers', type=int, default=4)  # 网络层数

    args = parser.parse_args([])

    # path = "./autodl-tmp"
    path = "."
    dataset_path = "data-bin/dataset{}.bin".format(args.sequence_length)
    model_save_path = path + "/model/celi.pth"

    # 词典
    bd = BidirectionalDict()
    bd.add_json_file('vocab.json')
    print("分词器加载完成")

    print("数据集路径为：{}".format(dataset_path))
    if os.path.exists(dataset_path):
        # 文件存在，加载数据集
        with open(dataset_path, 'rb') as f:
            word_list = pickle.load(f)
        dataset = WordDataset(args, bd, word_list)
        # 使用数据集进行后续操作
        print("成功加载数据集！")
    else:
        dataset = WordDataset(args, bd)
        array_size = sys.getsizeof(dataset.words_indexes)

        print("数组占用的内存大小为:", array_size, "字节")
        # 保存数据集到文件
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset.words_indexes, f)
        print("数据集首次加载完成,保存二进制文件完成")

    if os.path.exists(model_save_path):
        model = torch.load(model_save_path)
        print('发现有保存的Model,load model ....\n------开始训练----------')
    else:
        print('没保存的Model,Creat model .... \n------开始训练----------')
        model = BiLstmModel(13500, input_size, args.hidden_size, args.embedding_dim, args.num_layers, device)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的参数数量为: {total_params}")
    print(model)

    # 训练
    train(dataset, model, args)

    torch.save(model, model_save_path)

    print("训练完成")

    # 生成
    # dataset = None
    # save_pred_novel_path = "./novel/pred_novel_" + str(int(round(time.time() * 1000000))) + ".txt"
    # pred_novel_start_text = "慕娘子，你回来了？”"
    #
    # neirong = predict(bd, model, pred_novel_start_text, 1000)
    # print(neirong)
    #
    # for i in range(1, 30):
    #     pred_novel_start_text = '第' + str(i) + '章'
    #     neirong = predict(bd, model, pred_novel_start_text, 3000)
    #     with open(save_pred_novel_path, 'a+', buffering=1073741824, encoding='utf-8') as wf:
    #         wf.write(neirong)
