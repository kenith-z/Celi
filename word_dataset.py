import multiprocessing
from multiprocessing import Pool
import torch


import os

from util import encode_word





class WordDataset(torch.utils.data.Dataset):
    def __init__(self, args, bd, list = None):
        self.args = args
        self.bd = bd

        #把小说的 字 转换成 int
        self.words_indexes = []
        if list==None:
            words = self.load_words()
            # 创建一个进程池，设置进程数量
            num_threads = multiprocessing.cpu_count()
            print("当前CPU的线程数：", num_threads)
            pool = Pool(processes=num_threads)  
            
            # 并行地对单词进行编码
            results = pool.starmap(encode_word, [(self.bd, w) for w in words])

            # 关闭进程池
            pool.close()
            pool.join()

            # 将编码结果扁平化，并存储到self.words_indexes中
            self.words_indexes = [index for sublist in results for index in sublist]
        else:
            self.words_indexes = list


    def load_words(self):
        """加载数据集"""
        corpus_chars = ""
        print(self.args)
        file_list = os.listdir(self.args.train_novel_path)
        for filename in file_list:
            file_path = os.path.join(self.args.train_novel_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, encoding='UTF-8') as f:
                    novel_chars = f.read()
                    corpus_chars += novel_chars
        print('length', len(corpus_chars))
        print("本次加载：{}".format(file_list))
        return corpus_chars

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        
        return (
            torch.tensor(self.words_indexes[index:index + self.args.sequence_length]),
            torch.tensor(self.words_indexes[index + 1:index + self.args.sequence_length + 1]),
        )

