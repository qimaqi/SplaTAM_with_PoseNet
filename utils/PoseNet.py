import copy
import os
import time

import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as torch_F

class TransNet(torch.nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.input_t_dim = cfg['tracking']['poseNet_freq'] * 2 + 1
        self.define_network(cfg)
        self.device = cfg['tracking']['device']
        print("tracking self.device", self.device)
        self.max_index= cfg['n_img'] 
        # self.max_index.to(self.device )
        self.min_index = 0
        self.cfg = cfg

    def define_network(self,cfg):
        # input_dim dummy without encoding = 1
        # Pose translation prediction
        # torch.nn.init.zeros_(self.time_encoder.weight)
        # torch.nn.init.uniform_(self.time_encoder.weight,b=1e-6)
        self.mlp_transnet = torch.nn.ModuleList()
        layers_list = cfg['tracking']['layers_feat'] 
        L = list(zip(layers_list[:-1],layers_list[1:]))  


        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in cfg['tracking']['skip'] : k_in += self.input_t_dim
            if li==len(L)-1: k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_transnet.append(linear)

    def tensorflow_init_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, index):
        index = torch.tensor(index).to(self.device)
        # todo encoding the index
        index = index.reshape(-1,1).to(torch.float32)
        index = 2*(index - self.min_index)/(self.max_index - self.min_index) - 1

        points_enc = self.positional_encoding(index, L=self.cfg['tracking']['poseNet_freq'] )
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]

        translation_feat = points_enc
        activ_f = getattr(torch_F,'relu') 

        for li,layer in enumerate(self.mlp_transnet):
            if li in self.cfg['tracking']['skip']: translation_feat = torch.cat([translation_feat,points_enc],dim=-1)
            translation_feat = layer(translation_feat)
            if li==len(self.mlp_transnet)-1:
                translation_feat = torch_F.tanh(translation_feat)
            else:
                translation_feat = activ_f(translation_feat) 

        return translation_feat

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device = self.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc



class RotsNet(torch.nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.input_t_dim = cfg['tracking']['poseNet_freq'] * 2 + 1
        self.define_network(cfg)
        self.max_index = cfg['n_img'] 
        self.min_index = 0
        self.device = cfg['tracking']['device']

    def define_network(self,cfg):
        layers_list = cfg['tracking']['layers_feat'] 
        L = list(zip(layers_list[:-1],layers_list[1:]))  

        self.mlp_quad = torch.nn.ModuleList()
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in cfg['tracking']['skip']: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 4 
            linear = torch.nn.Linear(k_in,k_out)

            self.tensorflow_init_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_quad.append(linear)

    def tensorflow_init_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, index):
        index = torch.tensor(index).to(self.device)
        # todo encoding the index
        index = index.reshape(-1,1).to(torch.float32)
        activ_f = getattr(torch_F,'relu') 

        index = 2*(index - self.min_index)/(self.max_index - self.min_index) - 1

        points_enc = self.positional_encoding(index,L=self.cfg['tracking']['poseNet_freq'] )
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]        
        rotation_feat = points_enc

        for li,layer in enumerate(self.mlp_quad):
            if li in self.cfg['tracking']['skip']: rotation_feat = torch.cat([rotation_feat,points_enc],dim=-1)
            rotation_feat = layer(rotation_feat)
            if li==len(self.mlp_quad)-1:
                rotation_feat[:,1:] = torch_F.tanh(rotation_feat[:,1:])#torch_F.sigmoid(rotation_feat[:,1:])
                rotation_feat[:,0] = 1*(1 - torch_F.tanh(rotation_feat[:,0]))
            else:
                rotation_feat = activ_f(rotation_feat)


        norm_rots = torch.norm(rotation_feat,dim=-1)
        rotation_feat_norm = rotation_feat / (norm_rots[...,None] +1e-18)

        return rotation_feat_norm


    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device = self.device)*np.pi # [L] # ,device=cfg.device
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
    
