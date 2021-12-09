import torch
import numpy as np


def transform_embedding(embedding, low, high, operation):
    np.random.seed(22)
    torch.random.manual_seed(22)
    emb_size = embedding.shape[1]
    ran = torch.from_numpy(np.random.uniform(low, high,emb_size)).float().cuda()
    if operation == '+':
        return embedding + ran
    elif operation == '-':
        return embedding - ran

# def transform_embedding(embedding, low, high, operation):
#     np.random.seed(22)
#     torch.random.manual_seed(22)
#     emb_size = embedding.shape[1]
#     identity = torch.eye(emb_size)
#     tf = torch.from_numpy(np.random.uniform(low, high, emb_size)).float()
#     tf = tf * identity
#     if(operation == '+'):
#         tf = torch.sum(tf, dim=0)
#         result = embedding.cpu() + tf.cpu()
#     elif(operation == '*'):
#         result = torch.mm(embedding.cpu(), tf.cpu())
#     return result
    # return torch.mm(embedding.cpu(), tf.cpu())