#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os


def Identity(x):
    return x


class GammaIntersection(nn.Module):

    def __init__(self, dim):
        super(GammaIntersection, self).__init__()
        self.dim = dim
        self.layer_alpha1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_beta1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_alpha2 = nn.Linear(self.dim, self.dim)
        self.layer_beta2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_alpha1.weight)
        nn.init.xavier_uniform_(self.layer_beta1.weight)
        nn.init.xavier_uniform_(self.layer_alpha2.weight)
        nn.init.xavier_uniform_(self.layer_beta2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_alpha = F.relu(self.layer_alpha1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        attention1 = F.softmax(self.layer_alpha2(layer1_alpha), dim=0)  # (num_conj, batch_size, dim)

        layer1_beta = F.relu(self.layer_beta1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        attention2 = F.softmax(self.layer_beta2(layer1_beta), dim=0)  # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention1 * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention2 * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding


class GammaUnion(nn.Module):
    def __init__(self, dim, projection_regularizer, drop):
        super(GammaUnion, self).__init__()
        self.dim = dim
        self.layer_alpha1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_beta1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_alpha2 = nn.Linear(self.dim, self.dim // 2)
        self.layer_beta2 = nn.Linear(self.dim, self.dim // 2)
        self.layer_alpha3 = nn.Linear(self.dim // 2, self.dim)
        self.layer_beta3 = nn.Linear(self.dim // 2, self.dim)

        self.projection_regularizer = projection_regularizer
        self.drop = nn.Dropout(p=drop)
        nn.init.xavier_uniform_(self.layer_alpha1.weight)
        nn.init.xavier_uniform_(self.layer_beta1.weight)
        nn.init.xavier_uniform_(self.layer_alpha2.weight)
        nn.init.xavier_uniform_(self.layer_beta2.weight)
        nn.init.xavier_uniform_(self.layer_alpha3.weight)
        nn.init.xavier_uniform_(self.layer_beta3.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_alpha = F.relu(self.layer_alpha1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        layer2_alpha = F.relu(self.layer_alpha2(layer1_alpha))
        attention1 = F.softmax(self.drop(self.layer_alpha3(layer2_alpha)), dim=0)  # (num_conj, batch_size, dim)

        layer1_beta = F.relu(self.layer_beta1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        layer2_beta = F.relu(self.layer_beta2(layer1_beta))
        attention2 = F.softmax(self.drop(self.layer_beta3(layer2_beta)), dim=0)  # (num_conj, batch_size, dim)

        k = alpha_embeddings * attention1
        o = 1 / (beta_embeddings * attention2)
        k_sum = torch.pow(torch.sum(k * o, dim=0), 2) / torch.sum(torch.pow(o, 2) * k, dim=0)
        o_sum = torch.sum(k * o, dim=0) / (k_sum * o.shape[0])
        # Welch–Satterthwaite equation
        
        alpha_embedding = k_sum
        beta_embedding = o_sum
        alpha_embedding[torch.abs(alpha_embedding) < 1e-4] = 1e-4
        beta_embedding[torch.abs(beta_embedding) < 1e-4] = 1e-4
        return alpha_embedding, beta_embedding


class GammaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(GammaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

        self.layerr1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layerr2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layerr3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layerr0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layerr{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layerr{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, alpha_embedding, beta_embedding, alpha_embedding_r, beta_embedding_r):

        xa = torch.cat([alpha_embedding, alpha_embedding_r], dim=-1)
        xb = torch.cat([beta_embedding, beta_embedding_r], dim=-1)

        for nl in range(1, self.num_layers + 1):
            xa = F.relu(getattr(self, "layer{}".format(nl))(xa))
        xa = self.layer0(xa)
        xa = self.projection_regularizer(xa)
        for nl in range(1, self.num_layers + 1):
            xb = F.relu(getattr(self, "layerr{}".format(nl))(xb))
        xb = self.layerr0(xb)
        xb = self.projection_regularizer(xb)

        alpha_embeddings = xa
        beta_embeddings = xb

        return alpha_embeddings, beta_embeddings


class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None, gamma_mode=None, drop=0.):
        super(KGReasoning, self).__init__()

        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.is_u = False
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size,
                                                                               1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2))  # alpha and beta
        self.entity_regularizer = Regularizer(1, 0.15, 1e9)  # make sure the parameters of beta embeddings are positive
        self.projection_regularizer = Regularizer(1, 0.15,
                                                  1e9)  # make sure the parameters of beta embeddings after relation projection are positive

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-3 * self.embedding_range.item(),
            b=3 * self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-3 * self.embedding_range.item(),
            b=3 * self.embedding_range.item()
        )

        self.alpha_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.alpha_embedding,
            a=-3 * self.embedding_range.item(),
            b=3 * self.embedding_range.item()
        )

        self.beta_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.beta_embedding,
            a=-3 * self.embedding_range.item(),
            b=3 * self.embedding_range.item()
        )
        self.modulus = nn.Parameter(torch.Tensor([1 * self.embedding_range.item()]), requires_grad=True)

        hidden_dim, num_layers = gamma_mode
        self.center_net = GammaIntersection(self.entity_dim)
        self.projection_net = GammaProjection(self.entity_dim,
                                              self.relation_dim,
                                              hidden_dim,
                                              self.projection_regularizer,
                                              num_layers)
        self.union_net = GammaUnion(self.entity_dim, self.projection_regularizer, drop)

    def embed_query_gamma(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        if query_structure == ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)):
            aa = 1
        all_relation_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break

        if all_relation_flag:
            if query_structure[0] == 'e':
                ent_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                a_embedding, b_embedding = torch.chunk(ent_embedding, 2, dim=-1)
                idx += 1
                alpha_embedding = a_embedding
                beta_embedding = b_embedding
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_gamma(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    alpha_embedding = 1. / alpha_embedding
                    indicator_positive = beta_embedding >= 1
                    indicator_negative = beta_embedding < 1
                    beta_embedding[indicator_positive] = beta_embedding[indicator_positive] - 0.07
                    beta_embedding[indicator_negative] = beta_embedding[indicator_negative] + 0.07
                else:
                    alpha_r_embedding = torch.index_select(self.alpha_embedding, dim=0, index=queries[:, idx])
                    beta_r_embedding = torch.index_select(self.beta_embedding, dim=0, index=queries[:, idx])
                    alpha_embedding, beta_embedding = self.projection_net(alpha_embedding, beta_embedding,
                                                                          alpha_r_embedding, beta_r_embedding)
                idx += 1

        else:
            if self.is_u:
                alpha_embedding_list = []
                beta_embedding_list = []
                for i in range(len(query_structure)):
                    alpha_embedding, beta_embedding, idx = self.embed_query_gamma(queries, query_structure[i], idx)
                    alpha_embedding_list.append(alpha_embedding)
                    beta_embedding_list.append(beta_embedding)

                alpha_embedding, beta_embedding = self.union_net(torch.stack(alpha_embedding_list),
                                                                 torch.stack(beta_embedding_list))
            else:
                alpha_embedding_list = []
                beta_embedding_list = []
                for i in range(len(query_structure)):
                    alpha_embedding, beta_embedding, idx = self.embed_query_gamma(queries, query_structure[i], idx)
                    alpha_embedding_list.append(alpha_embedding)
                    beta_embedding_list.append(beta_embedding)
                alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                                  torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def cal_logit_gamma(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.gamma.Gamma(alpha_embedding, beta_embedding)
        distance = torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        logit = self.gamma - distance

        return logit

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []

        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                self.is_u = True
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_gamma(self.transform_union_query(batch_queries_dict[query_structure],
                                                                      query_structure),
                                           self.transform_union_structure(query_structure),
                                           0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                self.is_u = False
                alpha_embedding, beta_embedding, _ = self.embed_query_gamma(batch_queries_dict[query_structure],
                                                                            query_structure,
                                                                            0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.gamma.Gamma(all_alpha_embeddings, all_beta_embeddings)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_dists = torch.distributions.gamma.Gamma(all_union_alpha_embeddings, all_union_beta_embeddings)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[
                    all_idxs]
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_gamma(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[
                    all_union_idxs]
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(
                        1))

                positive_union_logit = self.cal_logit_gamma(positive_embedding, all_union_dists)
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(
                        batch_size, negative_size, -1))
                negative_logit = self.cal_logit_gamma(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(
                        batch_size, negative_size, -1))
                negative_union_logit = self.cal_logit_gamma(negative_embedding, all_union_dists)
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def transform_union_query(self, queries, query_structure):

        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([queries[:, :4], queries[:, 5:6]], dim=1)
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return (('e', ('r',)), ('e', ('r',)))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ((('e', ('r',)), ('e', ('r',))), ('r',))

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample,
                                                                      subsampling_weight, batch_queries_dict,
                                                                      batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader,
                                                                                      disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(
                        argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort,
                                               model.batch_entity_range)  # achieve the ranking of all entities
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(
                        zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics
