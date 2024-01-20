import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import estimate_koopman_matrix

class VAMPLoss(nn.Module):

    def __init__(self, epsilon=1e-6, mode='regularize', symmetrized=False):
        super(VAMPLoss, self).__init__()

        self._score = None
        self._score_list = []

        self._epsilon = epsilon
        self._mode = mode
        self._symmetrized = symmetrized

    def forward(self, data):
        assert len(data) == 2

        koopman = estimate_koopman_matrix(data[0], data[1], epsilon=self._epsilon, mode=self._mode, symmetrized=self._symmetrized)
        self._score = torch.pow(torch.norm(koopman, p='fro'), 2) + 1

        return -self._score
    
    def save(self):
        with torch.no_grad():
            self._score_list.append(self._score)
        return self

    def clear(self):
        self._score_list = []
        return self

    def output_mean_score(self):
        mean_score = torch.mean(torch.stack(self._score_list))
        return mean_score
    
class DisLoss(nn.Module):

    def __init__(self, feat_dim, n_states, device, proto_update_factor=0.5, scaling_temperature=0.1):
        super(DisLoss, self).__init__()

        self._score = None
        self._score_list = []

        self.register_buffer("prototypes", torch.zeros(n_states, feat_dim))
        self.n_states = n_states
        self.device = device
        self.proto_update_factor = proto_update_factor
        self.scaling_temperature = scaling_temperature
    
    def forward(self, features, labels):

        prototypes = self.prototypes.to(device=self.device)
        for i in range(len(labels)):
            prototypes[labels[i].item()] = F.normalize((self.proto_update_factor*prototypes[labels[i].item()] + (1-self.proto_update_factor)*features[i]), dim=0)
        self.prototypes = prototypes.detach()

        logits = torch.div(torch.matmul(prototypes,prototypes.T),self.scaling_temperature)
        proxy_labels = torch.arange(0, self.n_states).to(device=self.device)
        proxy_labels = proxy_labels.contiguous().view(-1, 1)
        mask = (1- torch.eq(proxy_labels, proxy_labels.T).float()).to(device=self.device)

        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = mean_prob_neg.mean()
        self._score = loss 
        
        return loss if not torch.isnan(loss) else 0
    
    def save(self):
        with torch.no_grad():
            self._score_list.append(self._score)
        return self

    def clear(self):
        self._score_list = []
        return self

    def output_mean_score(self):
        mean_score = torch.mean(torch.stack(self._score_list))
        return mean_score
    
class Prototypes(nn.Module):

    def __init__(self, n_states, device, scaling_temperature=0.1):
        super(Prototypes,self).__init__()

        self._proto_list = []
        self._disloss_list = []

        self.n_states = n_states
        self.device = device
        self.scaling_temperature = scaling_temperature

    def forward(self, features, labels):
        with torch.no_grad():
            proxy_labels = torch.arange(0, self.n_states).to(device=self.device)
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, proxy_labels).float().to(device=self.device)
            # [d,N] * [N,n_states]
            prototypes = torch.matmul(features.T, mask).T
            for i in range(self.n_states):
                # if prototypes.any() == 0:
                    # continue
                prototypes[i] = F.normalize(prototypes[i],dim=0)
    
            logits = torch.div(torch.matmul(prototypes,prototypes.T),self.scaling_temperature)
            proxy_labels = torch.arange(0, self.n_states).to(device=self.device)
            proxy_labels = proxy_labels.contiguous().view(-1, 1)
            mask = (1- torch.eq(proxy_labels, proxy_labels.T).float()).to(device=self.device)

            mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
            mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
            disloss = mean_prob_neg.mean()
            #disloss = disloss if not torch.isnan(disloss) else 0
            
            self._proto_list.append(prototypes)
            self._disloss_list.append(disloss)

        return prototypes

    def clear(self):
        self._proto_list = []
        self._disloss_list = []
        return self

    def output_mean_prototypes(self):
        mean_prototypes = torch.mean(torch.stack(self._proto_list),dim=0)
        for i in range(self.n_states):
            if mean_prototypes.any() == 0:
                continue
            mean_prototypes[i] = F.normalize(mean_prototypes[i],dim=0)
        return mean_prototypes
    
    def output_mean_disloss(self):
        mean_disloss = torch.mean(torch.stack(self._disloss_list))
        return mean_disloss if not torch.isnan(mean_disloss) else 0
