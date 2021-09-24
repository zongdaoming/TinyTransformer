#!/usr/bin/env python
#Author: Daoming Zong
#Date: 2021-09-22 15:47:23
#LastEditTime: 2021-09-22 15:47:25
#LastEditors: Daoming Zong and Chunya Liu
#Description: 
#FilePath: /models/SmallT/models/light_conditional/pointwise_feed_forward.py
#Copyright (c) 2021 SenseTime IRDC Group. All Rights Reserved.
"""Positionwise feed forward layer definition."""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))