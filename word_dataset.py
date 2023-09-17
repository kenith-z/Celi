import multiprocessing
from multiprocessing import Pool
import torch

import os

from util import encode_word


class WordDataset(torch.utils.data.Dataset):
    def __init__(self, args, bd, word_list=None):
        self.args = args
        self.bd = bd

        # 把小说的 字 转换成 int
        if word_list is None:
            words = self.load_words()
            self.words_indexes = self.parallel_encode(words)
        else:
            self.words_indexes = word_list

    def parallel_encode(self, words):
        num_threads = multiprocessing.cpu_count()
        print("当前CPU的线程数：", num_threads)

        with Pool(processes=num_threads) as pool:
            results = pool.starmap(encode_word, [(self.bd, w) for w in words])

        return [index for sublist in results for index in sublist]

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

        return (torch.tensor(self.words_indexes[index:index + self.args.sequence_length]),
                torch.tensor(self.words_indexes[index + 1:index + self.args.sequence_length + 1]))
