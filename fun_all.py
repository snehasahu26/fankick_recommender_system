# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:43:35 2020

@author: ssahu
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

def call_fun(user_id):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_interaction= pd.read_csv('new_interaction_data.csv',index_col=0)
    
    #selecting the coloumns
    new_interaction=new_interaction[["_id_x","_id_y","int_score"]]
    
    #rename the coloums as per dataset
    ratings = new_interaction.rename(columns = {"_id_x": "userId", 
                                      "_id_y":"movieId","int_score":"rating"})
    
    # getting the three column names from a pandas dataframe
    user_col, item_col, rating_col = ratings.columns
    
    # this function returns a python dictionary
    # which maps each id to a corresponding index value
    def list_2_dict(id_list:list):
        d={}
        for id, index in zip(id_list, range(len(id_list))):
            d[id] = index
        return d
    
    # splits ratings dataframe to training and validation dataframes
    def get_data(ratings, valid_pct:float = 0.2):
        # shuffle the indexes
        ln = random.sample(range(0, len(ratings)), len(ratings))
        
        # split based on the given validation set percentage 
        part = int(len(ln)*valid_pct)
        
        valid_index = ln[0:part]
        train_index = ln[part:]
        valid = ratings.iloc[valid_index]
        train = ratings.iloc[train_index]
        return [train,valid]
    
    
    # get a batch -> (user, item and rating arrays) from the dataframe
    def get_batch(ratings, start:int, end:int):
        return ratings[user_col][start:end].values, ratings[item_col][start:end].values, ratings[rating_col][start:end].values
    
    
    # get list of unique user ids
    users = sorted(list(set(ratings[user_col].values)))
    
    # get list of unique item ids
    items = sorted(list(set(ratings[item_col].values)))
    
    # generate dict of correponding indexes for the user ids
    user2idx = list_2_dict(users)
    
    # generate dict of correponding indexes for the item ids
    item2idx = list_2_dict(items)
    
    
    # neural net based on Embedding matrices
    class EmbeddingModel(nn.Module):
        def __init__(self, n_factors, n_users, n_items, y_range, initialise = 0.01):
            super().__init__()
            self.y_range = y_range
            self.u_weight = nn.Embedding(n_users, n_factors)
            self.i_weight = nn.Embedding(n_items, n_factors)
            self.u_bias = nn.Embedding(n_users, 1)
            self.i_bias = nn.Embedding(n_items, 1)
            
            # initialise the weights of the embeddings
            self.u_weight.weight.data.uniform_(-initialise, initialise)
            self.i_weight.weight.data.uniform_(-initialise, initialise)
            self.u_bias.weight.data.uniform_(-initialise, initialise)
            self.i_bias.weight.data.uniform_(-initialise, initialise)
    
        def forward(self, users, items):
            # dot multiply the weights for the given user_id and item_id
            dot = self.u_weight(users)* self.i_weight(items)
            
            # sum the result of dot multiplication above and add both the bias terms
            res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
            
            # return the output in the given range
            return torch.sigmoid(res) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]
        
    # create a model object
    # y_range has been extended(0-11) than required(1-10) to make the
    # values lie in the linear region of the sigmoid function
    model = EmbeddingModel(10, len(users), len(items), [0,11], initialise = 0.01).to(device)
    '''
    model_file = 'weights_only.pkl'
    torch.save(model.state_dict(), model_file)
    print("Model saved.")
    '''
    
    
    model_file = 'weights_only.pkl'
    model.load_state_dict(torch.load(model_file))
    print("New model created from saved weights")
    
    #model.eval()
    
    def recommend_item_for_user(model, user_id):
        m = model.eval().cpu()
        user_ids = torch.LongTensor([user2idx[u] for u in [user_id]*len(items)])
        item_ids = torch.LongTensor([item2idx[b] for b in items])
        remove = set(ratings[ratings[user_col] == user_id][item_col].values)
        preds = m(user_ids,item_ids).detach().numpy()
        pred_item = [(p,b) for p,b in sorted(zip(preds,items), reverse = True) if b not in remove]
        return pred_item
    
    #recommend_item_for_user3=recommend_item_for_user(model,"59bd3c43b137a16aac26f79c")
    
    
    # get recommendation 
    recommend_item_for_user3=recommend_item_for_user(model,user_id)
    recommend_item_for_user2=recommend_item_for_user3[0:10]
    recommend_item_for_user1=pd.DataFrame(recommend_item_for_user2)
        
    #ranaming the coloumns   
    recommend_item_for_user1 = recommend_item_for_user1.rename(columns = {0: "intrest_score", 
                                      1:"celeb_id"}) 
    
    #loading the celebrity feature data to map Celebrity id with Celebrity name
    Celeb_Features_df= pd.read_pickle("Celebrity_Features_df.pkl")
    Celeb_Features_df=Celeb_Features_df[["_id","celebrityName"]]
    
        
    out = (recommend_item_for_user1.merge(Celeb_Features_df, left_on='celeb_id', right_on='_id')
              .reindex(columns=['intrest_score','celeb_id', 'celebrityName']))
    
    #printing the result out 
    return out
    
#call_fun("59bd3c43b137a16aac26f79c")  