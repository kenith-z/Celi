import json

class BidirectionalDict:
    def __init__(self):
        self.forward_dict = {}# 正向索引字典，用于通过键获取值
        self.backward_dict = {}# 反向索引字典，用于通过值获取键
    
    def __len__(self):
            return len(self.forward_dict)

    def add(self, word, token):
        self.forward_dict[word] = token
        self.backward_dict[token] = word
    
    def get_token(self, word):
        return self.forward_dict.get(word)
    
    def get_word(self, token):
        return self.backward_dict.get(token)
    
    def add_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f ) # 从JSON文件中加载数据
        
        for word, token in data.items(): # 遍历JSON数据的键值对
            self.add(word, token) # 将键值对添加到双向索引字典中


# 示例用法
# if __name__ == '__main__':
#     bd = BidirectionalDict()
#     bd.add_json_file('vocab.json')
#     print(bd.get_token("a"))  
#     print(bd.get_word(42))    



