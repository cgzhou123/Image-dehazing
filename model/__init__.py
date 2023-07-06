'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-11-03 10:31:39
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-12-07 14:20:49
FilePath: /one_shot/model/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import torch
import torch.nn as nn
from importlib import import_module

class Model(nn.Module):
    def __init__(self, modelname):
        super(Model, self).__init__()
        
        print('Making model...')
        
        module = import_module('model.' + modelname)
        self.model = module.make_model()

    def forward(self, x, flag):
        return self.model(x, flag)
