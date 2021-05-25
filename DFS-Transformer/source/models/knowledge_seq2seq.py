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
import copy
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np
import time
from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.decoders.knowledge_enhanced_decoder import KnowledgeDecoder
from source.utils.criterions import NLLLoss
from source.utils.misc import Pack
from source.utils.misc import get_attn_key_pad_mask
from source.utils.misc import get_sinusoid_encoding_table
from source.utils.misc import get_subsequent_mask
from source.utils.misc import get_pos
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

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, d_model=512, inndim=1024, max_length=500,
                 padding_idx=None,
                 num_layers=4, nhead=4, attn_mode="general", activation=None, norm=False, attn_hidden_size=None,
                 with_bridge=False, tie_embedding=False, dropout=0.0, use_gpu=False, use_bow=False,
                 use_kd=False, use_dssm=False, use_posterior=False, weight_control=False,
                 use_pg=False, use_gs=False, pretrain_epoch=0):
        super(KnowledgeSeq2Seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.d_model = d_model
        self.inndim = inndim
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.nhead = nhead
        self.attn_mode = attn_mode
        self.activation = activation if activation else 'relu'
        self.norm = norm
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

        n_position = self.max_length + 1
        self.enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                     embedding_dim=self.embed_size, padding_idx=self.padding_idx)

        self.dim_trans = nn.Linear(in_features=self.embed_size, out_features=self.d_model)

        postinal_table = get_sinusoid_encoding_table(n_position, self.d_model, padding_idx=0)
        self.position_enc = nn.Embedding.from_pretrained(postinal_table, freeze=True)

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = self.enc_embedder
            self.knowledge_embedder = self.enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size, padding_idx=self.padding_idx)
            self.knowledge_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                               embedding_dim=self.embed_size,
                                               padding_idx=self.padding_idx)
        uttr_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                        nhead=self.nhead,
                                                        dim_feedforward=self.inndim,
                                                        dropout=self.dropout)
        self.uttr_encoder = nn.TransformerEncoder(encoder_layer=uttr_encoder_layer,
                                                  num_layers=self.num_layers,
                                                  norm=self.norm)
        know_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                        nhead=self.nhead,
                                                        dim_feedforward=self.inndim,
                                                        dropout=self.dropout)
        self.know_encoder = nn.TransformerEncoder(encoder_layer=know_encoder_layer,
                                                  num_layers=self.num_layers,
                                                  norm=self.norm)

        self.prior_attention = Attention(query_size=self.d_model,
                                         memory_size=self.d_model,
                                         hidden_size=self.d_model,
                                         mode="mlp")
        if self.use_posterior:
            self.posterior_attention = Attention(query_size=self.d_model * 2,
                                                 memory_size=self.d_model,
                                                 hidden_size=self.d_model,
                                                 mode="general")
        self.selection_attention = Attention(query_size=self.d_model,
                                             memory_size=2 * self.d_model,
                                             hidden_size=self.d_model,
                                             mode="general")
        self.knowledge_enhanced_decoder = KnowledgeDecoder(
            d_model=self.d_model,
            output_size=self.tgt_vocab_size,
            embed_size=self.embed_size,
            embedder=dec_embedder,
            position_enc=self.position_enc,
            dim_trans=self.dim_trans,
            num_layers=self.num_layers,
            nhead=self.nhead,
            attn_mode=self.attn_mode,
            activation=self.activation,
            norm=self.norm,
            dim_feedforward=self.inndim,
            dropout=self.dropout,
            tgt_emb_prj_weight_sharing=self.tgt_emb_prj_weight_sharing
        )

        if self.use_bow:
            self.bow_tgt_word_prj = nn.Linear(self.embed_size, self.tgt_vocab_size, bias=False)
            nn.init.xavier_normal_(self.bow_tgt_word_prj.weight)
            if self.tgt_emb_prj_weight_sharing:
                self.bow_tgt_word_prj.weight = self.enc_embedder.weight
                self.x_logit_scale = (self.d_model ** -0.5)
            else:
                self.x_logit_scale = 1
            self.bow_output_layer = nn.Sequential(
                nn.Linear(in_features=self.d_model, out_features=self.embed_size),
                nn.Tanh(),
                self.bow_tgt_word_prj,
            )
            self.softmax = nn.LogSoftmax(dim=-1)
        if self.use_kd:
            self.knowledge_dropout = nn.Dropout()

        if self.padding_idx is not None:
            self.weight = torch.ones(self.tgt_vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')
        self.knowledge_loss = NLLLoss(weight=None, reduction='mean')
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, is_training=False):
        """
        encode
        """
        '''
	    #inputs: 嵌套形式为{分离src和target和cue->(分离数据和长度->tensor数据值    
	    #{'src':( 数据值-->shape(batch_size,src_max_len), 数据长度值--> shape(batch_size) ),
	      'tgt':( 数据值-->shape(batch_size,src_max_len), 数据长度值-->shape(batch_size) )，
	      'cue' :( 数据值-->shape(batch_size,fact_size,cue_max_len), 数据长度值--> shape(batch_size,fact_size) )}
	    '''
        start_time = time.time()

        outputs = Pack()

        # --Prepare Message Sequence 
        src_seq, lengths = inputs.src[0][:, 1:-1], inputs.src[1] - 2  # lengths:(batch_size)

        # --Prepare Fatcs Sequence 
        batch_size, fact_size, cue_max_len = inputs.cue[0].size()
        facts_len = inputs.cue[1].view(-1)  # (batch_size*fact_size)
        cue_seq = inputs.cue[0].view(-1, cue_max_len)  # (batch_size*fact_size,cue_max_len)
        num_valid = facts_len.gt(0).int().sum().item()
        sorted_lengths, indices = facts_len.sort(descending=True)
        valid_cue_seq = cue_seq.index_select(0, indices)[:num_valid]  # (num_valid ,cue_max_len)

        # -- Prepare position sequence
        src_pos = get_pos(inputs.src[0]).type_as(inputs.src[0])[:, 1:-1]  # (batch_size,src_max_len)
        facts_len[facts_len > 0] -= 2
        valid_cue_pos = get_pos(valid_cue_seq).type_as(inputs.cue[0])[:, 1:-1]  # (num_valid , cue_max_len)
        valid_cue_seq = valid_cue_seq[:, 1:-1]
        # cue_pos = get_pos(cue_seq).type_as(inputs.cue[0])[:, 1:-1] # (batch_size*fact_size , cue_max_len)
        # cue_seq = cue_seq[:, 1:-1]
        # -- Prepare masks
        src_key_padding_mask = get_attn_key_pad_mask(seq_k=src_seq)  # (batch_size, src_max_len)
        valid_cue_key_padding_mask = get_attn_key_pad_mask(seq_k=valid_cue_seq)  # (batch_size*fact_size, cue_max_len)

        # -- Prepare embedding
        src_input = self.dim_trans(self.enc_embedder(src_seq)) + self.position_enc(
            src_pos)  # (batch_size,src_max_len,d_model)
        valid_cue_input = self.dim_trans(self.knowledge_embedder(valid_cue_seq)) + self.position_enc(
            valid_cue_pos)  # (batch_size*fact_size , cue_max_len , d_model)

        # -- Forward
        # src
        src_input = src_input.transpose(0, 1).contiguous()  # (src_max_len,batch_size,d_model)
        elapsed = time.time() - start_time
        print('编码前预处理time:', elapsed)
        start_time = time.time()
        src_outputs = self.uttr_encoder(src_input,
                                        mask=None,
                                        src_key_padding_mask=src_key_padding_mask)  # (src_max_len,batch_size,d_model)
        src_none_padding_idx = src_key_padding_mask.transpose(0, 1).contiguous()
        src_globals = src_outputs.masked_fill(src_none_padding_idx.unsqueeze(-1), 0.0).sum(0) / (
            lengths.unsqueeze(1).float())  # (batch_size,d_model)
        # src_globals = src_outputs.sum(0) / (lengths.unsqueeze(1).float())  # (batch_size,d_model)
        elapsed = time.time() - start_time
        print('uttr编码time:',elapsed)
        # facts
        valid_cue_input = valid_cue_input.transpose(0, 1).contiguous()  # (cue_max_len, num_valid, d_model)
        valid_cue_outputs = self.know_encoder(valid_cue_input,
                                              mask=None,
                                              src_key_padding_mask=valid_cue_key_padding_mask)  # (cue_max_len,num_valid,d_model)

        valid_cue_none_padding_idx = valid_cue_key_padding_mask.transpose(0, 1).contiguous()  # (cue_max_len, num_valid)
        norm_facts_len = copy.deepcopy(facts_len[:num_valid])  # (cue_max_len, num_valid)
        norm_facts_len[norm_facts_len == 0] += 1
        valid_cue_globals = valid_cue_input.masked_fill(valid_cue_none_padding_idx.unsqueeze(-1), 0.0).sum(0) / (
            norm_facts_len.unsqueeze(1).float())  # (num_valid , d_model)
        # valid_cue_globals = valid_cue_input.sum(0) / (norm_facts_len.unsqueeze(1).float())
        cue_outputs = valid_cue_outputs.new_zeros(size=(cue_max_len - 2, batch_size * fact_size, self.d_model),
                                                  dtype=torch.float)
        cue_globals = valid_cue_globals.new_zeros(size=(batch_size * fact_size, self.d_model), dtype=torch.float)
        cue_outputs[:, :num_valid, :] = valid_cue_outputs
        cue_globals[:num_valid, :] = valid_cue_globals

        _, inv_indices = indices.sort()
        cue_outputs = cue_outputs.index_select(1, inv_indices)  # (cue_max_len, batch_size*fact_size, d_model)
        cue_globals = cue_globals.index_select(0, inv_indices)  # (batch_size*fact_size, d_model)

        cue_outputs = cue_outputs.view(batch_size, fact_size, cue_max_len - 2,
                                       self.d_model)  # (batch_size, fact_size, cue_max_len, d_model)
        cue_globals = cue_globals.view(batch_size, fact_size, -1)  # (batch_size,fact_size,d_model)
        facts_len = facts_len.view(batch_size, fact_size)  # (batch_size, fact_size)

        # -- Knolwedge Optimazation

        # piror distribuation
        weighted_cue, cue_attn, _ = self.prior_attention(query=src_globals.unsqueeze(1),
                                                         memory=cue_globals,
                                                         mask=inputs.cue[1].eq(
                                                             0))  # weighted_cue:(batch_size , 1 , d_model)

        outputs.add(prior_attn=cue_attn)
        clues = weighted_cue  # (batch_size , 1 , d_model)

        if self.use_posterior:
            # Prepare target
            tgt_seq, tgt_lengths = inputs.tgt[0][:, 1:-1], inputs.tgt[1] - 2
            tgt_pos = get_pos(inputs.tgt[0]).type_as(inputs.tgt[0])[:, 1:-1]  # (batch_size,tgt_max_len)
            tgt_key_padding_mask = get_attn_key_pad_mask(seq_k=tgt_seq)  # (batch_size,tgt_max_len)
            tgt_input = self.dim_trans(self.knowledge_embedder(tgt_seq)) + self.position_enc(
                tgt_pos)  # (batch_size,tgt_max_len,d_model)
            tgt_input = tgt_input.transpose(0, 1).contiguous()  # (tgt__max_len,batch_size,d_model)
            tgt_outputs = self.know_encoder(tgt_input,
                                            mask=None,
                                            src_key_padding_mask=tgt_key_padding_mask)  # (tgt_max_len,batch_size,d_model)
            tgt_none_padding_idx = tgt_key_padding_mask.transpose(0, 1).contiguous()
            tgt_globals = tgt_outputs.masked_fill(tgt_none_padding_idx.unsqueeze(-1), 0).sum(0) / (
                tgt_lengths.unsqueeze(1).float())  # (batch_size,d_model)
            # tgt_globals = tgt_outputs.sum(0) / (tgt_lengths.unsqueeze(1).float())  # (batch_size,d_model)

            # postrior distribuation
            post_query = torch.cat([src_globals, tgt_globals], dim=-1)  # (batch_size , 2 * d_model)
            posterior_weighted_cue, posterior_attn, _ = self.posterior_attention(
                query=post_query.unsqueeze(1),
                memory=cue_globals,
                mask=inputs.cue[1].eq(0))
            # posterior_attn = posterior_attn.squeeze(1)  # cue_attn:(batch_size, fact_size)
            clues = posterior_weighted_cue  # (batch_size , 1 , d_model)
            outputs.add(posterior_attn=posterior_attn)

        optimized_knowledge = torch.cat([clues.repeat(1, fact_size, 1), cue_globals],
                                        dim=-1)  # (batch_size, fact_size ,2 * d_model)

        final_weighted_cue, final_cue_attn, knowledge_logits = self.selection_attention(query=src_globals.unsqueeze(1),
                                                                                        memory=optimized_knowledge,
                                                                                        mask=inputs.cue[1].eq(0))
        final_cue_attn = final_cue_attn.squeeze(1)  # (batch_size,fact_size)
        outputs.add(knowledge_logits=knowledge_logits)
        selection_indexs = final_cue_attn.max(dim=1)[1]  # (batch_size)

        if self.use_gs:
            if is_training:
                final_gumbel_attn = F.gumbel_softmax(torch.log(final_cue_attn + 1e-10), 0.1, hard=True)
                # selected_knowledge = torch.bmm(final_gumbel_attn.unsqueeze(1), cue_globals)
                selection_indexs = final_gumbel_attn.max(-1)[1]  # (batch_size)
        # if self.use_gs:
        #     if is_training:
        #         final_gumbel_attn = F.gumbel_softmax(torch.log(cue_attn + 1e-10), 0.1, hard=True)
        #         # selected_knowledge = torch.bmm(final_gumbel_attn.unsqueeze(1), cue_globals)
        #         selection_indexs = final_gumbel_attn.max(-1)[1]  # (batch_size)

        # choose the final fact
        fcor_enc_outputs = torch.gather(cue_outputs, 1,
                                        selection_indexs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                          cue_outputs.size(
                                                                                                              2),
                                                                                                          cue_outputs.size(
                                                                                                              3))).squeeze(
            1)  # (batch_size,cue_max_len,d_model)
        fcor_wordindex = torch.gather(inputs.cue[0][:, :, 1:-1], 1,
                                      selection_indexs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                          cue_outputs.size(2))).squeeze(
            1)  # (batch_size,cue_max_len)
        fcor_length = torch.gather(facts_len, 1, selection_indexs.unsqueeze(-1).repeat(1, 1)).squeeze(1)  # (batch_size)

        if self.use_bow:
            bow_logits = self.softmax(self.bow_output_layer(clues) * self.x_logit_scale)
            outputs.add(bow_logits=bow_logits)

        outputs.add(selection_indexs=selection_indexs)
        # outputs.add(reasoning_indexs=reasoning_indexs)
        if 'index' in inputs.keys():
            outputs.add(attn_index=inputs.index)

        # if self.use_kd:
        #     knowledge_hidden = self.knowledge_dropout(knowledge_hidden)

        dec_init_state = self.knowledge_enhanced_decoder.initialize_state(
            context_memory=src_outputs.transpose(0, 1).contiguous(),  # (src_max_len,batch_size,d_model)
            knowledge_memory=fcor_enc_outputs,  # (batch_size,cue_max_len,,d_model)
            attn_mode=self.attn_mode,
            memory_key_padding_mask=src_key_padding_mask,
            knowledge_length=fcor_length,
            memory_mask=None,
            knowledge_mask=None,
            knowledge_seq=fcor_wordindex,
            src_seq=src_seq)
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state = self.knowledge_enhanced_decoder.decode(input, state, is_generation=True)
        return log_prob, state

    def forward(self, enc_inputs, dec_inputs, is_training=False):
        """
        forward
        """
        outputs, dec_init_state = self.encode(
            enc_inputs, is_training=is_training)
        log_probs, _ = self.knowledge_enhanced_decoder(dec_inputs, dec_init_state)

        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, target, key_words, epoch=-1):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        bow = 0
        kl_loss = 0
        knowledge_loss = 0
        bias = 0.5

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
        if 'attn_index' in outputs:
            knowledge_target = outputs.attn_index  # (batch_size)
            knowledge_loss = self.knowledge_loss(knowledge_logits, knowledge_target)

            metrics.add(knowledge_loss=knowledge_loss)
        scores = -self.nll_loss(logits, target, reduction=False)

        nll_loss = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll_loss, num_words), acc=acc)

        if self.use_bow:
            bow_logits = outputs.bow_logits
            bow_labels = key_words  # (batch_size, key_num)
            bow_logits = bow_logits.repeat(1, bow_labels.size(-1), 1)
            bow = self.nll_loss(bow_logits, bow_labels)
            metrics.add(bow=bow)
        if self.use_posterior:
            kl_loss = self.kl_loss(torch.log(outputs.prior_attn + 1e-10), outputs.posterior_attn.detach())
            metrics.add(kl_loss=kl_loss)

        if epoch == -1:
            loss = nll_loss + knowledge_loss + bow + kl_loss
        elif epoch > self.pretrain_epoch:
            loss = nll_loss + bias * (knowledge_loss + kl_loss + bow)
        else:
            loss = knowledge_loss + bow + kl_loss

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

        return metrics, scores, logits

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
        metrics, scores, logits = self.collect_metrics(outputs, target, key_words, epoch=epoch)

        loss = metrics.loss
        if torch.isnan(loss):
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
            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()
            # optimizer.step()
        return metrics, scores
