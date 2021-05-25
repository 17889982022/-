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
from source.utils.misc import get_attn_key_pad_mask
from source.utils.misc import get_subsequent_mask
from source.utils.misc import get_pos
from source.modules.decoders.Layer import KnowledgeTransformerDecoder
from source.modules.decoders.Layer import KnowledgeTransformerDecoderLayer

from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class KnowledgeDecoder(nn.Module):
    """
    A HGFU GRU recurrent neural network decoder.
    Paper <<Towards Implicit Content-Introducing for Generative Short-Text
            Conversation Systems>>
    """

    def __init__(self,
                 d_model,
                 output_size,
                 embed_size,
                 embedder=None,
                 position_enc=None,
                 dim_trans=None,
                 num_layers=6,
                 nhead=8,
                 attn_mode=None,
                 activation='relu',
                 norm=None,
                 dim_feedforward=2048,
                 dropout=0.0,
                 tgt_emb_prj_weight_sharing=True):
        super(KnowledgeDecoder, self).__init__()

        self.d_model = d_model
        self.output_size = output_size
        self.embed_size = embed_size
        self.nhead = nhead
        self.attn_mode = attn_mode
        self.num_layers = num_layers
        self.activation = activation
        self.norm = norm
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.eps = 1e-12
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing

        decoder_layer = KnowledgeTransformerDecoderLayer(self.d_model,
                                                         self.nhead,
                                                         dim_feedforward=self.dim_feedforward,
                                                         dropout=self.dropout)
        self.decoder = KnowledgeTransformerDecoder(decoder_layer=decoder_layer, num_layers=self.num_layers,
                                                   norm=self.norm)

        self.dec_embedder = embedder
        self.position_enc = position_enc
        self.dim_trans = dim_trans

        if self.attn_mode is not None:
            self.attention_copy_x = Attention(query_size=self.d_model,
                                              memory_size=self.d_model,
                                              hidden_size=self.d_model * 2,
                                              mode=self.attn_mode,
                                              project=False)
            self.attention_copy_fact = Attention(query_size=self.d_model,
                                                 memory_size=self.d_model,
                                                 hidden_size=self.d_model * 2,
                                                 mode=self.attn_mode,
                                                 project=False)
            self.attention_copy_fusion = Attention(query_size=self.d_model,
                                                   memory_size=self.d_model,
                                                   hidden_size=self.d_model * 2,
                                                   mode=self.attn_mode,
                                                   project=False)
            self.fc1_copy = nn.Linear(self.d_model, 1)
            self.fc2_copy = nn.Linear(self.d_model, 1)
            self.fc3_copy = nn.Linear(self.d_model, 1)

        self.tgt_word_prj = nn.Linear(self.embed_size, self.output_size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if self.tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.dec_embedder.weight
            self.x_logit_scale = (self.d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.dec_embedder.weight.size(1), bias=False),
            nn.Dropout(p=self.dropout),
            self.tgt_word_prj)

    def initialize_state(self,
                         context_memory,
                         knowledge_memory,
                         memory_mask=None,
                         attn_mode=None,
                         memory_key_padding_mask=None,
                         knowledge_mask=None,
                         knowledge_length=None,
                         knowledge_seq=None,
                         src_seq=None
                         ):
        """
        initialize_state
        """
        if self.attn_mode is not None:
            assert context_memory is not None and knowledge_memory is not None

        if knowledge_length is not None:
            knowledge_max_len = knowledge_memory.size(1)  # (cue_len, batch_size, d_model)
            knowledge_attn_mask = sequence_mask(knowledge_length, knowledge_max_len).eq(0)  # (batch_size, max_len-2)

        init_state = DecoderState(
            context_memory=context_memory,
            knowledge_memory=knowledge_memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
            knowledge_mask=knowledge_mask,
            knowledge_key_padding_mask=knowledge_attn_mask,
            knowledge_wordindex=knowledge_seq,
            message_wordindex=src_seq,
        )
        return init_state

    def decode(self, tgt_seq, state, is_generation=False):  # 这里是每一个时间步执行一次，注意这里batch_size特指有效长度，即当前时间步无padding的样本数
        """
        decode
        """

        context_memory = state.context_memory.transpose(0, 1).contiguous()  # (src_len，batch_size, d_model)
        knowledge_memory = state.knowledge_memory.transpose(0, 1).contiguous()  # (src_len，batch_size, d_model)
        memory_key_padding_mask = state.memory_key_padding_mask  # (batch_size, src_len)
        knowledge_key_padding_mask = state.knowledge_key_padding_mask  # (batch_size, src_len)
        memory_mask = state.memory_mask  # None
        knowledge_mask = state.knowledge_mask  # None
        message_wordindex = state.message_wordindex
        knowledge_wordindex = state.knowledge_wordindex

        if is_generation:
            tgt_seq, tgt_pos = tgt_seq
        else:
            tgt_pos = get_pos(tgt_seq).type_as(tgt_seq)  # （batch_size, t)

        tgt_mask = get_subsequent_mask(tgt_seq)
        tgt_key_padding_mask = get_attn_key_pad_mask(seq_k=tgt_seq)

        tgt_input = self.dim_trans(self.dec_embedder(tgt_seq)) + self.position_enc(tgt_pos)  # (batch_size, t , d_model)
        tgt_input = tgt_input.transpose(0, 1).contiguous()  # (t, batch_size, d_model)
        decoder_outputs = self.decoder(tgt_input,
                                       memory=context_memory,
                                       knowledge=knowledge_memory,
                                       tgt_mask=tgt_mask,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       knowledge_mask=knowledge_mask,
                                       knowledge_key_padding_mask=knowledge_key_padding_mask)  # (t, batch_size, d_model)
        current_position_output = decoder_outputs[-1, :, :].unsqueeze(1)  # (batch_size, 1, d_model)
        decoder_outputs = decoder_outputs.transpose(0, 1).contiguous()  # (batch_size, tgt_len,  d_model)

        # Pointer Network

        query_copy = decoder_outputs  # (batch_size, tgt_len,  d_model)
        y_input = tgt_input.transpose(0, 1).contiguous()  # (batch_size, tgt_len,  d_model)
        dialg_context_copy, dialog_attn_cppy, _ = self.attention_copy_x(query=query_copy,
                                                                        memory=context_memory.transpose(0,
                                                                                                        1).contiguous(),
                                                                        # (batch_size , src_len , d_model)
                                                                        mask=memory_key_padding_mask)
        # dialg_context_copy(batch_size , tgt_len, d_model),dialog_attn_cppy(batch_size , tgt_len, src_len)
        fact_context_copy, fact_attn_cppy, _ = self.attention_copy_fact(query=query_copy,
                                                                        memory=knowledge_memory.transpose(0,
                                                                                                          1).contiguous(),
                                                                        mask=knowledge_key_padding_mask)  # fact_context_copy(batch_size, tgt_len, d_model)
        fusion_memory_copy = torch.cat([dialg_context_copy.unsqueeze(2), fact_context_copy.unsqueeze(2)],
                                       2)  # (batch_size , tgt_len, 2 , d_model)
        fusion_memory_copy = fusion_memory_copy.view(fusion_memory_copy.size(0) * fusion_memory_copy.size(1),
                                                     fusion_memory_copy.size(2), -1)

        fusion_context_copy, fusion_attn_cppy, _ = self.attention_copy_fusion(
            query=query_copy.view(query_copy.size(0) * query_copy.size(1), -1).unsqueeze(1),
            memory=fusion_memory_copy,
            mask=None)
        fusion_context_copy = fusion_context_copy.view(query_copy.size(0), query_copy.size(1), 1,
                                                       self.d_model)  # (batch_size , tgt_len, 1 , d_model)
        fusion_attn_cppy = fusion_attn_cppy.view(query_copy.size(0), query_copy.size(1), 1,
                                                 2)  # (batch_size , tgt_len, 1 , 2)
        dialog_weight = fusion_attn_cppy.squeeze(2)[:, :, 0]  # (batch_size,tgt_len)
        fact_weight = fusion_attn_cppy.squeeze(2)[:, :, 1]  # (batch_size,tgt_len)
        dialog_copy_p = torch.mul(dialog_attn_cppy, dialog_weight.unsqueeze(2))  # (batch_size,tgt_len,src_len)
        fact_copy_p = torch.mul(fact_attn_cppy, fact_weight.unsqueeze(2))  # (batch_size,tgt_len, cue_len)
        weight_gen = self.sigmoid(
            self.fc1_copy(fusion_context_copy.squeeze(2)) + self.fc2_copy(query_copy) + self.fc3_copy(
                y_input))  # (batch_size,tgt_len,1)

        logits = self.output_layer(decoder_outputs) * self.x_logit_scale
        log_probs_gen = self.softmax(logits)  # softmax输出置信度:(batch_size, tgt_len, tgt_vacb_size)
        log_probs_dialog = log_probs_gen.new_zeros(size=(decoder_outputs.size(0), decoder_outputs.size(1),
                                                         self.output_size), dtype=torch.float)
        # （batch_size, tgt_len, vocab_size）
        log_probs_fact = log_probs_gen.new_zeros(size=(decoder_outputs.size(0), decoder_outputs.size(1),
                                                       self.output_size), dtype=torch.float)
        # （batch_size, tgt_len, vocab_size）

        log_probs_dialog.scatter_(2, message_wordindex.unsqueeze(1).repeat(1, log_probs_dialog.size(1), 1),
                                  dialog_copy_p)  # (batch_size,  tgt_len, vocab_size)
        log_probs_fact.scatter_(2, knowledge_wordindex.unsqueeze(1).repeat(1, log_probs_fact.size(1), 1),
                                fact_copy_p)  # (batch_size, tgt_len, vocab_size)

        log_probs_copy = log_probs_dialog + log_probs_fact
        log_probs = torch.mul(log_probs_gen, weight_gen) + torch.mul(log_probs_copy, 1 - weight_gen)
        log_probs = torch.log(log_probs + self.eps)

        return log_probs, state  # out_input： 要输入给为decoder的输出层；  state：decoder隐层状态

    def forward(self, inputs, state):
        """
        forward
        """

        tgt_seq, lengths = inputs  # inputs:(batch_size，tgt_len-1)“tgt”去尾 , lengths(batch_size)
        log_probs, state = self.decode(tgt_seq, state, is_generation=False)
        # decoder_output: (batch_size, tgt_len,  d_model)
        # dialog_copy_p : (batch_size,tgt_len, src_len)
        # fact_copy_p : (batch_size, tgt_len, cue_len)
        # weight_gen : (batch_size,tgt_len,1)

        return log_probs, state
