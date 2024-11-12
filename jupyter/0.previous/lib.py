import pickle
import torch
ws=pickle.load(open("../model/ws.pkl", "rb"))
max_len=200
hidden_size=512
num_layer=3
bidriectional=True
dropout=0.4
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=128
test_batch_size=4