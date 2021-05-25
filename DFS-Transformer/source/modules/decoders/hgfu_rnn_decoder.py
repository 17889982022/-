#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/decoders/hgfu_rnn_decoder.py
"""

import torch
import torch.nn as nn

from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState



from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class RNNDecoder(nn.Module):
    """
    A HGFU GRU recurrent neural network decoder.
    Paper <<Towards Implicit Content-Introducing for Generative Short-Text
            Conversation Systems>>
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0,
                 concat=False):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2
        self.memory_size = memory_size or hidden_size
        self.feature_size = feature_size
        self.dropout = dropout
        self.concat = concat
        self.eps = 1e-12

        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_size
        self.cue_input_size = self.hidden_size
        
        if self.feature_size is not None:
            self.rnn_input_size += self.feature_size
            self.cue_input_size += self.feature_size

        if self.attn_mode is not None:
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.attention_copy_x = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.attention_copy_fact = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.attention_copy_fusion = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            # self.cue_attention = Attention(query_size=self.hidden_size,
            #                            memory_size=self.memory_size,
            #                            hidden_size=self.attn_hidden_size,
            #                            mode=self.attn_mode,
            #                            project=False)
            self.rnn_input_size += self.memory_size
            self.cue_input_size += self.memory_size
            self.out_input_size += self.memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        self.cue_rnn = nn.GRU(input_size=self.cue_input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=self.dropout if self.num_layers > 1 else 0,
                              batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc1_copy = nn.Linear(self.hidden_size, 1)
        self.fc2_copy = nn.Linear(self.hidden_size, 1)
        self.fc3_copy = nn.Linear(self.input_size, 1)

        if self.concat:
            self.fc3 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.fc3 = nn.Linear(self.hidden_size * 2, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax()

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                # nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )

    def initialize_state(self,
                         hidden,
                         feature=None,
                         attn_memory=None,
                         attn_mask=None,
                         memory_lengths=None,
                         knowledge=None,
                         fcor=None,
                         fcor_mask=None,
                         fcor_length=None,
                         dialog_wordindex = None,
                         fcor_wordindex = None
                         ):
        """
        initialize_state
        """
        if self.feature_size is not None:
            assert feature is not None

        if self.attn_mode is not None:
            assert attn_memory is not None

        if memory_lengths is not None and attn_mask is None:
            max_len = attn_memory.size(1)#(batch_size, max_len-2, 2*rnn_hidden_size)
            attn_mask = sequence_mask(memory_lengths, max_len).eq(0)#(batch_size, max_len-2)

        if fcor_length is not None and fcor_mask is None:
            fcor_max_len = fcor.size(1)#(batch_size, max_len-2, 2*rnn_hidden_size)
            fcor_attn_mask = sequence_mask(fcor_length, fcor_max_len).eq(0)#(batch_size, max_len-2)

        init_state = DecoderState(
            hidden=hidden,
            feature=feature,
            attn_memory=attn_memory,
            attn_mask=attn_mask,
            knowledge=knowledge,
            fcor=fcor,
            fcor_mask=fcor_attn_mask,
            dialog_wordindex = dialog_wordindex,
            fcor_wordindex = fcor_wordindex
        )
        return init_state

    def decode(self, input, state, is_training=False):#这里是每一个时间步执行一次，注意这里batch_size特指有效长度，即当前时间步无padding的样本数
        """
        decode
        """
        


        hidden = state.hidden#上一个时间步的hidden,如果是第一个时间步则为enc_hidden: (1，batch_size, 2*rnn_hidden_size)
        rnn_input_list = []
        cue_input_list = []
        out_input_list = []#为decoder的输出层做准备
        output = Pack()

        if self.embedder is not None:
            input = self.embedder(input) # (batch_size, 1,  input_size)

       
        input = input.unsqueeze(1)         #(batch_size , input_size) 
        rnn_input_list.append(input)
        cue_input_list.append(state.knowledge)#knowledge：(batch_size, 1 , 2*rnn_hidden_size)

        if self.feature_size is not None:
            feature = state.feature.unsqueeze(1)
            rnn_input_list.append(feature)
            cue_input_list.append(feature)

        if self.attn_mode is not None:#对enc_hidden作attention
            attn_memory = state.attn_memory
            attn_mask = state.attn_mask
            query = hidden[-1].unsqueeze(1)
            # cue_query = state.knowledge
            weighted_context, attn,_= self.attention(query=query,
                                                    memory=attn_memory,
                                                    mask=attn_mask)#weighted_context(batch_size , 1 , 2*rnn_hidden_size)
            
            # cue_weighted_context, cue_attn,_ = self.cue_attention(query=query,
            #                                         memory=cue_memory,
            #                                         mask=cue_mask)#weighted_context(batch_size , 1 , 2*rnn_hidden_size)
            rnn_input_list.append(weighted_context)
            # rnn_input_list.append(cue_weighted_context)
            cue_input_list.append(weighted_context)
            # cue_input_list.append(cue_weighted_context)
            out_input_list.append(weighted_context)
            # out_input_list.append(cue_weighted_context)
            output.add(attn=attn)

        rnn_input = torch.cat(rnn_input_list, dim=-1)#(batch_size, 1 ,input_size + 2*rnn_hidden_size)
        rnn_output, rnn_hidden = self.rnn(rnn_input, hidden)#rnn_hidden(1, batch_size , 2*rnn_hidden_size)

        cue_input = torch.cat(cue_input_list, dim=-1)#(batch_size, 1 , 4*rnn_hidden_size)
        cue_output, cue_hidden = self.cue_rnn(cue_input, hidden)#cue_hidden(1, batch_size , 2*rnn_hidden_size)

        h_y = self.tanh(self.fc1(rnn_hidden))
        h_cue = self.tanh(self.fc2(cue_hidden))
        if self.concat:
            new_hidden = self.fc3(torch.cat([h_y, h_cue], dim=-1))#(1, batch_size , 2*rnn_hidden_size)
        else:
            k = self.sigmoid(self.fc3(torch.cat([h_y, h_cue], dim=-1)))
            new_hidden = k * h_y + (1 - k) * h_cue
        out_input_list.append(new_hidden[-1].unsqueeze(0).transpose(0, 1))

        out_input = torch.cat(out_input_list, dim=-1)#(batch_size, 1 , 3*hidden_size)这里是要输入给为decoder的输出层的，相当于c+h
        state.hidden = new_hidden#(1, batch_size , 2*rnn_hidden_size)为下一个时间步更新hidden


        #Pointer Network
        query_copy = new_hidden[-1].unsqueeze(1)
        dialg_context_copy, dialog_attn_cppy,_= self.attention_copy_x(query=query_copy,
                                                    memory=attn_memory,
                                                    mask=attn_mask)#dialg_context_copy(batch_size , 1 , 2*rnn_hidden_size),dialog_attn_cppy(batch_size , 1 , maxlen)
        
        fact_context_copy,fact_attn_cppy ,_ = self.attention_copy_fact(query=query_copy,
                                                    memory=state.fcor,
                                                    mask=state.fcor_mask)#fact_context_copy(batch_size , 1 , 2*rnn_hidden_size)

        fusion_memory_copy = torch.cat([dialg_context_copy,fact_context_copy],1)#(batch_size , 2 , 2*rnn_hidden_size)

        fusion_context_copy, fusion_attn_cppy,_= self.attention_copy_fusion(query=query_copy,
                                                    memory=fusion_memory_copy,
                                                    mask=None)#fact_context_copy(batch_size , 1 , 2*rnn_hidden_size)fusion_attn_cppy(batch_size , 1 , 2)
        dialog_weight = fusion_attn_cppy.squeeze(1)[:,0]#(batch_size)
        fact_weight = fusion_attn_cppy.squeeze(1)[:,1]
       
        dialog_copy_p = torch.mul(dialog_attn_cppy.squeeze(1),dialog_weight.unsqueeze(1))#(batch_size,maxlen)
        fact_copy_p = torch.mul(fact_attn_cppy.squeeze(1),fact_weight.unsqueeze(1))#(batch_size,maxlen)
        weight_gen = self.sigmoid(self.fc1_copy(fusion_context_copy)+self.fc2_copy(query_copy)+self.fc3_copy(input)).squeeze(1)#(batch_size,1)
        
        # dialog_copy_p = torch.mul(dialog_copy_p,weight_gen)
        # fact_copy_p = torch.mul(fact_copy_p,weight_gen)




        if is_training:
            return out_input, state, output,dialog_copy_p,fact_copy_p,weight_gen#out_input： 要输入给为decoder的输出层；  state：decoder隐层状态;  output:一个pack字典，包含key"attn"
        else:
            log_prob_gen = self.output_layer(out_input)
            log_prob_dialog = log_prob_gen.new_zeros(size=(log_prob_gen.size(0), 1, self.output_size),dtype=torch.float)#（batch_size, 1, vocab_size）
            log_prob_fact = log_prob_gen.new_zeros(size=(log_prob_gen.size(0),1 , self.output_size),dtype=torch.float)#（batch_size, 1, vocab_size）
            log_prob_dialog.scatter_(2,state.dialog_wordindex.unsqueeze(1),dialog_copy_p.unsqueeze(1))#(batch_size, 1, vocab_size)
            log_prob_fact.scatter_(2,state.fcor_wordindex.unsqueeze(1),fact_copy_p.unsqueeze(1))#(batch_size, 1, vocab_size)
            log_prob_copy = log_prob_dialog+log_prob_fact

            log_prob = torch.mul(log_prob_gen,weight_gen.unsqueeze(1)) + torch.mul(log_prob_copy,1-weight_gen.unsqueeze(1))
            log_prob = torch.log(log_prob + self.eps)


            return log_prob, state, output

    def forward(self, inputs, state,dialog_wordindex,fcor_wordindex):
        """
        forward
        """
        inputs, lengths = inputs#inputs:(batch_size，max_len-1)“tgt”去尾 , lengths(batch_size) 
        batch_size, max_len = inputs.size()
        x_maxlen = dialog_wordindex.size(1)#（batch_size, x_maxlen）
        f_maxlen = fcor_wordindex.size(1)
        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),dtype=torch.float)#（batch_size, tgt_max_len, 4*rnn_hidden_size）
        out_probs_copy_x = inputs.new_zeros(
            size=(batch_size, max_len, x_maxlen),dtype=torch.float)#（batch_size, tgt_max_len, x_maxlen）
        out_probs_copy_f = inputs.new_zeros(
            size=(batch_size, max_len, f_maxlen),dtype=torch.float)#（batch_size, tgt_max_len, f_maxlen）
        weight_gens = inputs.new_zeros(size=(batch_size, max_len, 1),dtype=torch.float)#（batch_size, tgt_max_len, 1）


        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)#更改en_hidden的batch顺序，使其与tgt长度排序顺序相同

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)#(max_len-1)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            out_input, valid_state, _ ,dialog_copy_p,fact_copy_p,weight_gen = self.decode(
                dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden

            out_inputs[:num_valid, i] = out_input.squeeze(1) #（batch_size, max_len, 4*rnn_hidden_size）
            out_probs_copy_x[:num_valid,i] = dialog_copy_p#(num_valid,x_maxlen)
            out_probs_copy_f[:num_valid,i] = fact_copy_p
            weight_gens[:num_valid,i] = weight_gen

        # Resort 撤回排序
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)
        out_probs_copy_x = out_probs_copy_x.index_select(0, inv_indices)
        out_probs_copy_f = out_probs_copy_f.index_select(0, inv_indices)
        weight_gens = weight_gens.index_select(0, inv_indices)#（batch_size, max_len, 1）







        log_probs_gen = self.output_layer(out_inputs)#softmax输出置信度:(batch_size, max_len, tgt_vacb_size)

        log_probs_dialog = log_probs_gen.new_zeros(size=(batch_size, max_len, self.output_size),dtype=torch.float)#（batch_size, max_len, vocab_size）
        log_probs_fact = log_probs_gen.new_zeros(size=(batch_size, max_len, self.output_size),dtype=torch.float)#（batch_size, max_len, vocab_size）
        
        log_probs_dialog.scatter_(2,dialog_wordindex.unsqueeze(1).repeat(1,max_len,1),out_probs_copy_x)#(batch_size,  max_len, vocab_size)
        log_probs_fact.scatter_(2,fcor_wordindex.unsqueeze(1).repeat(1,max_len,1),out_probs_copy_f)#(batch_size, max_len, vocab_size)
        
        log_probs_copy = log_probs_dialog+log_probs_fact
        log_probs = torch.mul(log_probs_gen,weight_gens) + torch.mul(log_probs_copy,1-weight_gens)
        log_probs = torch.log(log_probs+self.eps)





        
        return log_probs, state
