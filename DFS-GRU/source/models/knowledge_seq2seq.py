#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/knowledge_seq2seq.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.decoders.hgfu_rnn_decoder import RNNDecoder
from source.utils.criterions import NLLLoss
from source.utils.misc import Pack
from source.utils.metrics import accuracy
from source.utils.metrics import attn_accuracy
from source.utils.metrics import perplexity
from source.modules.attention import Attention
import torchsnooper
from source.utils.misc import sequence_mask
class KnowledgeSeq2Seq(BaseModel):
    """
    KnowledgeSeq2Seq
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, padding_idx=None,
                 num_layers=1, bidirectional=True, attn_mode="general", attn_hidden_size=None, 
                 with_bridge=False, tie_embedding=False, dropout=0.0, use_gpu=False, use_bow=False,
                 use_kd=False, use_dssm=False, use_posterior=False, weight_control=False, 
                 use_pg=False, use_gs=False, concat=False, pretrain_epoch=0):
        super(KnowledgeSeq2Seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.attn_hidden_size = attn_hidden_size
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_bow = use_bow
        self.use_dssm = use_dssm
        self.weight_control = weight_control
        self.use_kd = use_kd
        self.use_pg = use_pg
        self.use_gs = use_gs
        self.use_posterior = use_posterior
        self.pretrain_epoch = pretrain_epoch
        self.baseline = 0
        self.tgt_emb_prj_weight_sharing = True

        self.enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size, padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  embedder=self.enc_embedder, num_layers=self.num_layers,
                                  bidirectional=self.bidirectional, dropout=self.dropout)
        # self.selection_encoder = RNNEncoder(input_size=self.embed_size, hidden_size=self.hidden_size,
        #                           embedder=enc_embedder, num_layers=self.num_layers,
        #                           bidirectional=self.bidirectional, dropout=self.dropout)

        # self.bridge = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.Tanh())

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = self.enc_embedder
            knowledge_embedder = self.enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size, padding_idx=self.padding_idx)
            knowledge_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                          embedding_dim=self.embed_size,
                                          padding_idx=self.padding_idx)

        self.knowledge_encoder = RNNEncoder(input_size=self.embed_size,
                                            hidden_size=self.hidden_size,
                                            embedder=knowledge_embedder,
                                            num_layers=self.num_layers,
                                            bidirectional=self.bidirectional,
                                            dropout=self.dropout)
        
        self.prior_attention = Attention(query_size=self.hidden_size,
                                         memory_size=self.hidden_size,
                                         hidden_size=self.hidden_size,
                                         mode="mlp")
        if self.use_posterior:
            self.posterior_attention = Attention(query_size=self.hidden_size*2,
                                             memory_size=self.hidden_size,
                                             hidden_size=self.hidden_size,
                                             mode="general")
       
        self.tanh = nn.Tanh()

        self.reasoning_attention = Attention(query_size=self.hidden_size,
                                             memory_size=self.hidden_size*2,
                                             hidden_size=self.hidden_size,
                                             mode="general")


        self.decoder = RNNDecoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  output_size=self.tgt_vocab_size, embed_size = self.embed_size ,embedder=dec_embedder,
                                  num_layers=self.num_layers, attn_mode=self.attn_mode,
                                  memory_size=self.hidden_size, feature_size=None,
                                  dropout=self.dropout, concat=concat)
        # self.log_softmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        if self.use_bow:
            self.bow_tgt_word_prj = nn.Linear(self.embed_size, self.tgt_vocab_size, bias=False)
            nn.init.xavier_normal_(self.bow_tgt_word_prj.weight)
            if self.tgt_emb_prj_weight_sharing:
                self.bow_tgt_word_prj.weight = self.enc_embedder.weight
                self.x_logit_scale = (self.hidden_size ** -0.5)
            self.bow_output_layer = nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.embed_size),
                nn.Tanh(),
                self.bow_tgt_word_prj,
                )
            self.softmax = nn.LogSoftmax(dim=-1)

        if self.use_dssm:
            self.dssm_project = nn.Linear(in_features=self.hidden_size,
                                          out_features=self.hidden_size)
            self.mse_loss = torch.nn.MSELoss(reduction='mean')

        if self.use_kd:
            self.knowledge_dropout = nn.Dropout()

        if self.padding_idx is not None:
            self.weight = torch.ones(self.tgt_vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')
        self.knowledge_loss = NLLLoss(weight=None,reduction='mean')
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, hidden=None, is_training=False):
        """
        encode
        """
        '''
	    #inputs: 嵌套形式为{分离src和target和cue->(分离数据和长度->tensor数据值    
	    #{'src':( 数据值-->shape(batch_size,max_len), 数据长度值--> shape(batch_size) ),
	      'tgt':( 数据值-->shape(batch_size,max_len), 数据长度值-->shape(batch_size) )，
	      'cue' :( 数据值-->shape(batch_size,sent_num,max_len), 数据长度值--> shape(batch_size,sen_num) )}
	    '''
        outputs = Pack()
        #enc_inputs:((batch_size，max_len-2), (batch_size-2))**src去头去尾
        #hidden:None
        enc_inputs = _, lengths = inputs.src[0][:, 1:-1], inputs.src[1]-2
        enc_outputs, enc_hidden = self.encoder(enc_inputs, hidden)
        # enc_outputs_selection, enc_hidden = self.selection_encoder (enc_inputs, hidden)
        #enc_outputs:(batch_size, max_len-2, hidden_size)
        #enc_hidden:(1 , batch_size , hidden_size)
        
        # if self.with_bridge:
        #     enc_hidden = self.bridge(enc_hidden)

        # knowledge
        batch_size, sent_num, sent  = inputs.cue[0].size()
        tmp_len = inputs.cue[1]
        tmp_len[tmp_len > 0] -= 2
        cue_inputs = inputs.cue[0].view(-1, sent)[:, 1:-1], tmp_len.view(-1)
        #cue_inputs:((batch_size*sent_num , max_len-2),(batch_size*sent_num))
        cue_enc_outputs, cue_enc_hidden = self.knowledge_encoder(cue_inputs, hidden)
        #cue_enc_outputs:(batch_size*sent_num , max_len-2, hidden_size)
        #cue_enc_hidden:(1 , batch_size*sent_num, 2 * rnn_hidden_size)
        cue_outputs = cue_enc_hidden[-1].view(batch_size, sent_num, -1)
        #cue_outputs:(batch_size, sent_num, 2 * rnn_hidden_size)
        # Attention

        weighted_cue, cue_attn, _ = self.prior_attention(query=enc_hidden[-1].unsqueeze(1),
                                                      memory=cue_outputs,
                                                      mask=inputs.cue[1].eq(0))
        #weighted_cue:(batch_size , 1 , 2 * rnn_hidden_size)
        cue_attn = cue_attn.squeeze(1)
        #cue_attn:(batch_size, sent_num)
        outputs.add(prior_attn=cue_attn)
        #indexs:(batch_size)
        indexs = cue_attn.max(dim=1)[1]

        clues = weighted_cue#(batch_size , 1 , 2 * rnn_hidden_size)



        if self.use_posterior:
            tgt_enc_inputs = inputs.tgt[0][:, 1:-1], inputs.tgt[1]-2
            _, tgt_enc_hidden = self.knowledge_encoder(tgt_enc_inputs, hidden)
            intermediate_query = torch.cat([enc_hidden[-1].unsqueeze(1),tgt_enc_hidden[-1].unsqueeze(1)],dim=-1)#(batch_size ,1, hidden_size)
            # intermediate_query = torch.add(enc_hidden[-1].unsqueeze(1),tgt_enc_hidden[-1].unsqueeze(1))
            posterior_weighted_cue, posterior_attn, _ = self.posterior_attention(
                query=intermediate_query,
                memory=cue_outputs,
                mask=inputs.cue[1].eq(0))
            posterior_attn = posterior_attn.squeeze(1)
            outputs.add(posterior_attn=posterior_attn)
            clues = posterior_weighted_cue#(batch_size ,1, hidden_size)
            indexs = posterior_attn.max(dim=1)[1]

        # optimized_knowledge = torch.add(clues.repeat(1,cue_outputs.size(1),1), cue_outputs)
        optimized_knowledge = torch.cat([clues.repeat(1,cue_outputs.size(1),1), cue_outputs], dim = -1 )






       
        reasoning_weighted_cue, reasoning_cue_attn,knowledge_logits = self.reasoning_attention(query=enc_hidden[-1].unsqueeze(1),
                                                            memory=optimized_knowledge,
                                                            mask=inputs.cue[1].eq(0))
        reasoning_cue_attn = reasoning_cue_attn.squeeze(1)#(batch_size,sennum)
        outputs.add(knowledge_logits=knowledge_logits)
        reasoning_indexs = reasoning_cue_attn.max(dim=1)[1]


        if self.use_gs:
            if is_training:
                reasoning_gumbel_attn = F.gumbel_softmax(torch.log(reasoning_cue_attn + 1e-10), 0.1, hard=True)
                reasoning_knowledge = torch.bmm(reasoning_gumbel_attn.unsqueeze(1), cue_outputs)
                reasoning_indexs = reasoning_gumbel_attn.max(-1)[1]#(batch_size)
            else:
                reasoning_knowledge = cue_outputs.gather(1, reasoning_indexs.view(-1, 1, 1).repeat(1, 1, cue_outputs.size(-1)))
        else:
            reasoning_knowledge = reasoning_weighted_cue



        cue_enc_outputs = cue_enc_outputs.view(batch_size,cue_outputs.size(1),cue_enc_outputs.size(1),-1)#cue_enc_outputs(batch_size,sen_num,maxlen,hidden_size)
        fcor_enc_outputs = torch.gather(cue_enc_outputs,1,reasoning_indexs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,cue_enc_outputs.size(2),cue_enc_outputs.size(3))).squeeze(1)
        #fcor_enc_outputs(batch_size,maxlen,hidden_size)
        fcor_wordindex = torch.gather(inputs.cue[0][:, :,1:-1],1,reasoning_indexs.unsqueeze(-1).unsqueeze(-1).repeat(1,1,inputs.cue[0][:, :,1:-1].size(2))).squeeze(1)#fcor_wordindex(batch_size,maxlen)
        fcor_length =  torch.gather(tmp_len,1,reasoning_indexs.unsqueeze(-1).repeat(1,1)).squeeze(1)#fcor_enc_length(batch_size)
       
        knowledge_hidden = reasoning_knowledge
        # final_knowledge = torch.cat([knowledge,reasoning_knowledge],dim=-1)
        # knowledge_hidden = self.bridge(final_knowledge)
        if self.use_bow:
                bow_logits = self.softmax(self.bow_output_layer(clues)*self.x_logit_scale)

                outputs.add(bow_logits=bow_logits)

        outputs.add(indexs=reasoning_indexs)
        # outputs.add(reasoning_indexs=reasoning_indexs)
        if 'index' in inputs.keys():
            outputs.add(attn_index=inputs.index)

        if self.use_kd:
            knowledge_hidden = self.knowledge_dropout(knowledge_hidden)

        if self.weight_control:
            weights = (enc_hidden[-1] * knowledge_hidden.squeeze(1)).sum(dim=-1)
            weights = self.sigmoid(weights)
            # norm in batch
            # weights = weights / weights.mean().item()
            outputs.add(weights=weights)
            knowledge_hidden = knowledge_hidden * weights.view(-1, 1, 1).repeat(1, 1, knowledge_hidden.size(-1))

        dec_init_state = self.decoder.initialize_state(
            hidden=enc_hidden,
            attn_memory=enc_outputs if self.attn_mode else None,
            memory_lengths=lengths if self.attn_mode else None,
            knowledge=knowledge_hidden,
            fcor=fcor_enc_outputs,
            fcor_length=fcor_length,
            fcor_wordindex = fcor_wordindex,
            dialog_wordindex = inputs.src[0][:, 1:-1])
        outputs.add(fcor_wordindex = fcor_wordindex)
        outputs.add(dialog_wordindex = inputs.src[0][:, 1:-1])
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs, hidden=None, is_training=False):
        """
        forward
        """
        outputs, dec_init_state = self.encode(
                enc_inputs, hidden, is_training=is_training)
        fcor_wordindex = outputs.fcor_wordindex
        dialog_wordindex = outputs.dialog_wordindex
        log_probs, _ = self.decoder(dec_inputs, dec_init_state,dialog_wordindex,fcor_wordindex)
        
        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, target, key_words, epoch=-1):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        if epoch == -1 or epoch > self.pretrain_epoch:
            bias=0.6
        else:
            bias=1

        # test begin
        # nll = self.nll(torch.log(outputs.posterior_attn+1e-10), outputs.attn_index)
        # loss += nll
        # attn_acc = attn_accuracy(outputs.posterior_attn, outputs.attn_index)
        # metrics.add(attn_acc=attn_acc)
        # metrics.add(loss=loss)
        # return metrics
        # test end

        logits = outputs.logits
        knowledge_logits = outputs.knowledge_logits
        # print("print(knowledge_logits.size())",knowledge_logits.size())
        # print("logits",logits.size())
        if 'attn_index' in outputs:
            knowledge_target = outputs.attn_index#(batch_size)
            knowledge_loss = self.knowledge_loss(knowledge_logits, knowledge_target)
            loss += bias*knowledge_loss
            metrics.add(knowledge_loss = knowledge_loss)
        scores = -self.nll_loss(logits, target, reduction=False)

        nll_loss = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll_loss, num_words), acc=acc)
        
        if self.use_bow:
            bow_logits = outputs.bow_logits
            bow_labels = key_words
            bow_logits = bow_logits.repeat(1, bow_labels.size(-1), 1)
            bow = self.nll_loss(bow_logits, bow_labels)
            loss += bias*bow
            metrics.add(bow=bow)
        
        if self.use_posterior:
            kl_loss = self.kl_loss(torch.log(outputs.prior_attn + 1e-10),outputs.posterior_attn.detach())
            metrics.add(kl_loss=kl_loss)
            loss += bias * kl_loss

        if epoch==-1 or epoch >self.pretrain_epoch:
            loss += nll_loss


            
        # if self.use_posterior:
        #     kl_loss = self.kl_loss(torch.log(outputs.prior_attn + 1e-10),
        #                            outputs.posterior_attn.detach())
        #     metrics.add(kl=kl_loss)
        #     if self.use_bow:
        #         bow_logits = outputs.bow_logits
        #         bow_labels = target[:, :-1]
        #         bow_logits = bow_logits.repeat(1, bow_labels.size(-1), 1)
        #         bow = self.nll_loss(bow_logits, bow_labels)
        #         loss += bow
        #         metrics.add(bow=bow)
        #     if self.use_dssm:
        #         mse = self.mse_loss(outputs.dssm, outputs.reply_vec.detach())
        #         loss += mse
        #         metrics.add(mse=mse)
        #         pos_logits = outputs.pos_logits
        #         pos_target = torch.ones_like(pos_logits)
        #         neg_logits = outputs.neg_logits
        #         neg_target = torch.zeros_like(neg_logits)
        #         pos_loss = F.binary_cross_entropy_with_logits(
        #                 pos_logits, pos_target, reduction='none')
        #         neg_loss = F.binary_cross_entropy_with_logits(
        #                 neg_logits, neg_target, reduction='none')
        #         loss += (pos_loss + neg_loss).mean()
        #         metrics.add(pos_loss=pos_loss.mean(), neg_loss=neg_loss.mean())

        #     if epoch == -1 or epoch > self.pretrain_epoch or \
        #        (self.use_bow is not True and self.use_dssm is not True):
        #         loss += nll_loss
        #         loss += kl_loss
        #         if self.use_pg:
        #             posterior_probs = outputs.posterior_attn.gather(1, outputs.indexs.view(-1, 1))
        #             reward = -perplexity(logits, target, self.weight, self.padding_idx) * 100
        #             pg_loss = -(reward.detach()-self.baseline) * posterior_probs.view(-1)
        #             pg_loss = pg_loss.mean()
        #             loss += pg_loss
        #             metrics.add(pg_loss=pg_loss, reward=reward.mean())
        metrics.add(loss=loss)
        if 'attn_index' in outputs:
            attn_acc = attn_accuracy(outputs.knowledge_logits, outputs.attn_index)
            metrics.add(attn_acc=attn_acc)
        # else:
        
        return metrics, scores,logits

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=False, epoch=-1):
        """
        iterate
        """
        '''
	    #inputs: 嵌套形式为{分离src和target和cue->(分离数据和长度->tensor数据值    
	    #{'src':( 数据值-->shape(batch_size,max_len), 数据长度值--> shape(batch_size) ),
	      'tgt':( 数据值-->shape(batch_size,max_len), 数据长度值-->shape(batch_size) )，
	      'cue' :( 数据值-->shape(batch_size,sent_num,max_len), 数据长度值--> shape(batch_size,sen_num) )}
	    '''
        enc_inputs = inputs
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1
        target = inputs.tgt[0][:, 1:]
        key_words = inputs.key[0]

        outputs = self.forward(enc_inputs, dec_inputs, is_training=is_training)
        metrics, scores ,logits = self.collect_metrics(outputs, target, key_words,epoch=epoch)

        loss = metrics.loss
        if torch.isnan(loss):
            print("nan loss occoured")
            parm={}
            for name,parameters in self.named_parameters():
                parm[name] = parameters
            print(parm["reasoning_attention.linear_query.weight"])
            # print("logits",logits)
            raise ValueError("nan loss encountered")

        if is_training:
            if self.use_pg:
                self.baseline = 0.99 * self.baseline + 0.01 * metrics.reward.item()
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step_and_update_lr()
        return metrics, scores
