import argparse
import pprint
import os
import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from tsdart.utils import set_random_seed
from tsdart.model import TSDART, TSDARTLayer, TSDARTModel, TSDARTEstimator
from tsdart.dataprocessing import Preprocessing

parser = argparse.ArgumentParser(description='Training with TS-DART')

parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--device', default='cpu', type=str, help='train the model with gpu or cpu')

parser.add_argument('--lag_time', type=int, help='the lag time used to create transition pairs', required=True)

parser.add_argument('--encoder_sizes', nargs='+', type=int, help='the size of each layer in TS-DART encoder, the size of the last layer represents feat_dim', required=True)
parser.add_argument('--feat_dim', type=int, help='the dimensionality of latent space ((d-1)-hypersphere)', required=True)
parser.add_argument('--n_states', type=int, help='the number of metastable states to consider', required=True)

parser.add_argument('--beta', default=0.01, type=float, help='the weight of dispersion loss')
parser.add_argument('--gamma', default=1, type=float, help='the radius of hypersphere')
parser.add_argument('--proto_update_factor', default=0.5, type=float, help='the update factor to compute state center vectors in EMA algorithm')
parser.add_argument('--scaling_temperature', default=0.1, type=float, help='the scaling factor in despersion loss')

parser.add_argument('--optimizer', default='Adam', type=str, help='the optimizer to train the model')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='the learning rate to training the model')

parser.add_argument('--pretrain', default=10, type=int, help='the number of pretraining epochs with pure VAMP-2 loss optimization')
parser.add_argument('--n_epochs', default=20, type=int, help='the total number of training epochs with VAMP-2 and dispersion loss optimization')
parser.add_argument('--save_model_interval', default=None, type=int, help='save the model every save_epoch')

parser.add_argument('--train_split', default=0.9, type=float, help='the ratio of training dataset size to full dataset size')
parser.add_argument('--train_batch_size', default=1000, type=int, help='the batch size in training dataloader')
parser.add_argument('--val_batch_size', default=None, type=int, help='the batch size in validation dataloader')

parser.add_argument('--data_directory', type=str, help='the directory storing numpy files of trajectories', required=True)
parser.add_argument('--saving_directory', default='.', type=str, help='the saving directory of training results')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%m_%d_%H_%M")

args.name = (f"{date_time}_tsdart_lr_{args.learning_rate}_bsz_{args.train_batch_size}_"
        f"lag_time_{args.lag_time}_beta_{args.beta}_feat_dim_{args.feat_dim}_n_states_{args.n_states}_"
        f"pretrain_{args.pretrain}_n_epochs_{args.n_epochs}")

args.log_directory = args.saving_directory+"/{name}/logs".format(name=args.name)
args.model_directory = args.saving_directory+"/{name}/checkpoints".format(name=args.name)

if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)

with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

def main():

    device = torch.device(args.device)

    data = []
    np_name_list = []
    for np_name in glob.glob(args.data_directory+'/*.npy'):
        data.append(np.load(np_name))
        np_name_list.append(np_name.rsplit('/')[-1])

    set_random_seed(args.seed)

    pre = Preprocessing(dtype=np.float32)
    dataset = pre.create_dataset(lag_time=args.lag_time,data=data)

    val = int(len(dataset)*(1-args.train_split))
    train_data, val_data = random_split(dataset, [len(dataset)-val, val])

    loader_train = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    if val == 0:
        loader_val = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=False)
    else:
        if args.val_batch_size is None or args.val_batch_size >= len(val_data):
            loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        else:
            loader_val = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False)

    lobe = TSDARTLayer(args.encoder_sizes,n_states=args.n_states)
    lobe = lobe.to(device=device)

    tsdart = TSDART(lobe=lobe, learning_rate=args.learning_rate, device=device, beta=args.beta, feat_dim=args.feat_dim, n_states=args.n_states, 
                    pretrain=args.pretrain, save_model_interval=args.save_model_interval)
    tsdart_model = tsdart.fit(loader_train, n_epochs=args.n_epochs, validation_loader=loader_val).fetch_model()

    validation_vamp = tsdart.validation_vamp
    validation_dis = tsdart.validation_dis
    validation_prototypes = tsdart.validation_prototypes

    training_vamp = tsdart.training_vamp
    training_dis = tsdart.training_dis

    np.save((args.model_directory+'/validation_vamp.npy'),validation_vamp)
    np.save((args.model_directory+'/validation_dis.npy'),validation_dis)
    np.save((args.model_directory+'/validation_prototypes.npy'),validation_prototypes)

    np.save((args.model_directory+'/training_vamp.npy'),training_vamp)
    np.save((args.model_directory+'/training_dis.npy'),training_dis)

    if args.save_model_interval is None:
        torch.save(tsdart_model.lobe.state_dict(), args.model_directory+'/model_{}epochs.pytorch'.format(args.n_epochs))

        hypersphere_embs = tsdart_model.transform(data=data,return_type='hypersphere_embs')
        metastable_states = tsdart_model.transform(data=data,return_type='states')
        softmax_probs = tsdart_model.transform(data=data,return_type='probs')

        tsdart_estimator = TSDARTEstimator(tsdart_model)
        ood_scores = tsdart_estimator.fit(data).ood_scores
        state_centers = tsdart_estimator.fit(data).state_centers

        dir1 = args.model_directory+'/model_{}epochs_hypersphere_embs'.format(args.n_epochs)
        dir2 = args.model_directory+'/model_{}epochs_metastable_states'.format(args.n_epochs)
        dir3 = args.model_directory+'/model_{}epochs_softmax_probs'.format(args.n_epochs)
        dir4 = args.model_directory+'/model_{}epochs_ood_scores'.format(args.n_epochs)
        dir5 = args.model_directory+'/model_{}epochs_state_centers'.format(args.n_epochs)

        if not os.path.exists(dir1):
            os.makedirs(dir1)
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        if not os.path.exists(dir3):
            os.makedirs(dir3)
        if not os.path.exists(dir4):
            os.makedirs(dir4)
        if not os.path.exists(dir5):
            os.makedirs(dir5)

        np.save((dir5+'/state_centers.npy'),state_centers)

        if len(np_name_list) == 1: ### hypersphere_embs etc. is numpy array
            np.save((dir1+'/hypersphere_embs_'+np_name_list[0]),hypersphere_embs)
            np.save((dir2+'/metastable_states_'+np_name_list[0]),metastable_states)
            np.save((dir3+'/softmax_probs_'+np_name_list[0]),softmax_probs)
            np.save((dir4+'/ood_scores_'+np_name_list[0]),ood_scores)
        else:
            for k in range(len(np_name_list)): ### hypersphere_embs etc. is list of numpy arrays
                np.save((dir1+'/hypersphere_embs_'+np_name_list[k]),hypersphere_embs[k])
                np.save((dir2+'/metastable_states_'+np_name_list[k]),metastable_states[k])
                np.save((dir3+'/softmax_probs_'+np_name_list[k]),softmax_probs[k])
                np.save((dir4+'/ood_scores_'+np_name_list[k]),ood_scores[k])

    else:
        for i in range(len(tsdart._save_models)):
            torch.save(tsdart._save_models[i].lobe.state_dict(), args.model_directory+'/model_{}epochs.pytorch'.format((i+1)*args.save_model_interval))

            hypersphere_embs = tsdart._save_models[i].transform(data=data,return_type='hypersphere_embs')
            metastable_states = tsdart._save_models[i].transform(data=data,return_type='states')
            softmax_probs = tsdart._save_models[i].transform(data=data,return_type='probs')

            tsdart_estimator = TSDARTEstimator(tsdart._save_models[i])
            ood_scores = tsdart_estimator.fit(data).ood_scores
            state_centers = tsdart_estimator.fit(data).state_centers

            dir1 = args.model_directory+'/model_{}epochs_hypersphere_embs'.format((i+1)*args.save_model_interval)
            dir2 = args.model_directory+'/model_{}epochs_metastable_states'.format((i+1)*args.save_model_interval)
            dir3 = args.model_directory+'/model_{}epochs_softmax_probs'.format((i+1)*args.save_model_interval)
            dir4 = args.model_directory+'/model_{}epochs_ood_scores'.format((i+1)*args.save_model_interval)
            dir5 = args.model_directory+'/model_{}epochs_state_centers'.format((i+1)*args.save_model_interval)

            if not os.path.exists(dir1):
                os.makedirs(dir1)
            if not os.path.exists(dir2):
                os.makedirs(dir2)
            if not os.path.exists(dir3):
                os.makedirs(dir3)
            if not os.path.exists(dir4):
                os.makedirs(dir4)
            if not os.path.exists(dir5):
                os.makedirs(dir5)

            np.save((dir5+'/state_centers.npy'),state_centers)

            if len(np_name_list) == 1: ### hypersphere_embs etc. is numpy array
                np.save((dir1+'/hypersphere_embs_'+np_name_list[0]),hypersphere_embs)
                np.save((dir2+'/metastable_states_'+np_name_list[0]),metastable_states)
                np.save((dir3+'/softmax_probs_'+np_name_list[0]),softmax_probs)
                np.save((dir4+'/ood_scores_'+np_name_list[0]),ood_scores)
            else:
                for k in range(len(np_name_list)): ### hypersphere_embs etc. is list of numpy arrays
                    np.save((dir1+'/hypersphere_embs_'+np_name_list[k]),hypersphere_embs[k])
                    np.save((dir2+'/metastable_states_'+np_name_list[k]),metastable_states[k])
                    np.save((dir3+'/softmax_probs_'+np_name_list[k]),softmax_probs[k])
                    np.save((dir4+'/ood_scores_'+np_name_list[k]),ood_scores[k])

if __name__ == '__main__':
    main()