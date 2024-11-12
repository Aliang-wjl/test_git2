import os
class word :
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    UNK = 0
    PAD = 1
    def __init__(self):
        '''
        1、特殊字符存没有出现的词语
        2、对短句子进行填充
        '''
        self.dict={
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.count={}
    def fit(self,sentence):
        '''
        把单个句子保存到词典中去
        '''
        for word in sentence:
            self.count[word]=self.count.get(word,0)+1
    def build_vocab(self,min=5,max=None,max_features=None):
        """
        生成词典
        :param min: 词语最小出现的次数
        :param max: 词语最大出现的次数
        :param max_features: 一共保留多少个词语
        :return:
        """
        self.count={word:value for word,value in self.count.items() if value>min}#当他的次数大于min时候进行保留
        self.count = {word: value for word, value in self.count.items() if value <max}#当他的次数小于max进行保留
        sorted(self.count.items(),key=lambda x:x[1],reverse=True)[:max_features]
        for ward in self.count:
            self.dict[ward]=len(self.dict)
        #得到一个翻转的dict的字典
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))
    def transfrom(self,sentence,max_len=None):
        """
        将句子转换为一个序列
        :param sentence:
        :return:
        """
        if len(sentence) is not max_len:
            if max_len > len(sentence):
                sentence=sentence+[self.PAD_TAG]*(max_len-len(sentence))
            if max_len < len(sentence):
                sentence=sentence[:max_len]
        return [self.dict.get(ward,self.UNK) for ward in sentence]
    def inverse_transfrom(self,indices):
        """
        将序列转换成一个句子
        :param indices:
        :return:
        """
        return [self.inverse_dict.get(idx) for idx in indices]
    def __len__(self):
        return len(self.dict)
 
 
 
 
 
 