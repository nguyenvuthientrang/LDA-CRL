from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from networks.model import Encoder
from utils import Moment, dot_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from utils import osdist


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.id2cls = None
        self.cls2id = None


    def get_proto(self, args, encoder, mem_set):
        # aggregate the prototype set for further use.
        data_loader = get_data_loader(args, mem_set, False, False, 1)

        features = []

        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature, rep= encoder.bert_forward(tokens)
            features.append(feature)
            self.lbs.append(labels.item())
        features = torch.cat(features, dim=0)

        proto = torch.mean(features, dim=0, keepdim=True)

        return proto, features


    # Use K-Means to select what samples to save, similar to at_least = 0
    def select_data(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens=torch.stack([x.to(args.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())

        features = np.concatenate(features)
        num_clusters = min(args.num_protos, len(sample_set))
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

        mem_set = []
        current_feat = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
            current_feat.append(features[sel_index])
        
        current_feat = np.stack(current_feat, axis=0)
        current_feat = torch.from_numpy(current_feat)
        return mem_set, current_feat, current_feat.mean(0)
    
    def get_optimizer(self, args, encoder):
        print('Use {} optim!'.format(args.optim))
        def set_param(module, lr, decay=0):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ['undecay']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize
        params = set_param(encoder, args.learning_rate)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer

    def train_simple_model(self, args, encoder, training_data, epochs):

        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()

        optimizer = self.get_optimizer(args, encoder)

        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                hidden, reps = encoder.bert_forward(tokens)

                #train by contrastive loss
                loss = self.moment.loss(reps, labels)
                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()

                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach())
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(data_loader, "train_{}".format(epoch_i), is_mem=False)
    

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, test_data, protos4eval):
        data_loader = get_data_loader(args, test_data, batch_size=1)
        encoder.eval()
        n = len(test_data)
        correct = 0
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            labels = labels.to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            hidden, reps = encoder.bert_forward(tokens)

            logits = -osdist(hidden, protos4eval)
            seen_sim = logits.cpu().data.numpy()
            max_smi = np.max(seen_sim,axis=1)
            label_smi = logits[:,labels].cpu().data.numpy()
            if label_smi >= max_smi:
                correct += 1
        return correct/n

    def train(self, args):
        # set training batch
        for i in range(args.total_round):
            # set random seed
            random.seed(args.seed+i*100)


            # sampler setup
            sampler = data_sampler(args=args, seed=args.seed+i*100)
            self.id2cls = sampler.id2cls
            self.cls2id = sampler.cls2id
            # encoder setup
            encoder = Encoder(args=args).to(args.device)

            #data setup
            train, valid, test = sampler.training_dataset, sampler.valid_dataset, sampler.test_dataset
            training = {}
            validing = {}
            testing = {}

            for cls, id in self.cls2id.items():
                training[cls] = train[id]
                validing[cls] = valid[id]
                testing[cls] = test[id]

            # Initial
            training_data = []
            for cls in self.id2cls:
                training_data += training[cls]

            # initialize memory and prototypes
            num_class = len(sampler.id2cls)
            memorized_samples = {}            

            # train model
            self.moment = Moment(args)
            self.moment.init_moment(args, encoder, training_data, is_memory=False)
            self.train_simple_model(args, encoder, training_data, args.nepochs)

            feat_mem = []
            proto_mem = []

            for cls in self.id2cls:
                memorized_samples[cls], feat, temp_proto = self.select_data(args, encoder, training[cls])
                feat_mem.append(feat)
                proto_mem.append(temp_proto)

            feat_mem = torch.cat(feat_mem, dim=0)
            temp_proto = torch.stack(proto_mem, dim=0)

            protos4eval = temp_proto.to(args.device)

            test_data = []
            for cls in self.id2cls:
                test_data += testing[cls]

            cur_acc = self.evaluate_strict_model(args, encoder, test_data, protos4eval)

            print(f'current test acc:{cur_acc}')
            
            del self.moment