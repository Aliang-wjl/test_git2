#1、完成数据集的准备
from torch.utils.data import  Dataset,DataLoader
import os
import re
from lib import ws,max_len
import torch
import lib
def tokenlize(content):#分词
    tokens=[]
    fileters=['\t','\n','\x97','\x96','#','<.*?>']
    content=re.sub("|".join(fileters),' ',content)#为什么要用|.join我也不是很清楚
    tokens.extend(i.strip() for i in content.split())
    return  tokens
class ImdDataset(Dataset):
    def __init__(self,train=True):
        self.train_data_path= r"E:/python eye/project2/emation/aclImdb/train"
        self.test_data_path= r"E:/python eye/project2/emation/aclImdb/test"
        data_path=self.train_data_path if train else self.test_data_path
        #把所有文件名放入列表
        temp_data_path=[os.path.join(data_path,"pos"),os.path.join(data_path,"neg")]
        self.total_file_path=[]#所有评论的文件
        for path in temp_data_path:
            file_name_path=os.listdir(path)
            file_path_list=[os.path.join(path,i) for  i in file_name_path if i.endswith(".txt")]
            self.total_file_path.extend(file_path_list)
    def __getitem__(self, index):#有这个方法你才可以用
        file_path= self.total_file_path[index]
        file_path="".join(file_path)
        label_str=file_path.split('\\')[-2]
        label=0 if label_str =="neg" else 1
        #获取内容
        content=tokenlize(open(file_path,encoding='utf-8').read())
        return content,label
    def __len__(self):
        return len(self.total_file_path)
def collate_fn(batch):
    content,label=list(zip(*batch))
    content=[ws.transfrom(i,max_len=max_len)for i in content]
    content=torch.LongTensor(content)
    label=torch.LongTensor(label)
    return  content,label
def get_dataloader(train=True,batch_size=lib.batch_size):
    imdb_dataset=ImdDataset(train)
    data_loader=DataLoader(imdb_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    return data_loader
# for idx,(input,target) in enumerate(get_dataloader()):
    # print(idx)
    # print(input)
    # print(target)
    # break
 
 
 
 
 
 
 