import pickle
import random
import json, os
from transformers import AutoTokenizer
import numpy as np 

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    return tokenizer

class data_sampler(object):
    def __init__(self, args, seed=None):
        self.args = args
        temp_name = [args.dataname, args.seed]
        file_name = "{}.pkl".format("-".join([str(x) for x in temp_name]))

        self.save_data_path = os.path.join("./dat/", file_name)

        if not os.path.exists("./dat/"):
            os.mkdir("./dat/")

        self.tokenizer = get_tokenizer()

        # read class data
        self.id2cls, self.cls2id = self._read_classes(args.class_file)

        # random sampling
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2cls)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.args.data_file)

        self.batch = 0


        # record classes
        self.seen_classes = []
        self.history_test_data = {}

    def _read_data(self, file):
        if os.path.isfile(self.save_data_path):
            with open(self.save_data_path, 'rb') as f:
                datas = pickle.load(f)
            train_dataset, val_dataset, test_dataset = datas
            return train_dataset, val_dataset, test_dataset    

        else:
            data = json.load(open(file, 'r', encoding='utf-8'))
            train_dataset = [[] for i in range(self.args.num_classes)]
            val_dataset = [[] for i in range(self.args.num_classes)]
            test_dataset = [[] for i in range(self.args.num_classes)]
            for cls in data.keys():
                cls_samples = data[cls]
                if self.seed != None:
                    random.seed(self.seed)
                random.shuffle(cls_samples)
                for i, sample in enumerate(cls_samples):
                    tokenized_sample = {}
                    tokenized_sample['cls'] = self.cls2id[sample['cls']]
                    tokenized_sample['tokens'] = self.tokenizer.encode(sample['text'],
                                                                    padding='max_length',
                                                                    truncation=True,
                                                                    max_length=self.args.max_length)
                    if i < self.args.num_of_train*len(cls_samples):
                        train_dataset[self.cls2id[cls]].append(tokenized_sample)
                    elif i < (self.args.num_of_train + self.args.num_of_val)*len(cls_samples):
                        val_dataset[self.cls2id[cls]].append(tokenized_sample)
                    else:
                        test_dataset[self.cls2id[cls]].append(tokenized_sample)
            with open(self.save_data_path, 'wb') as f:
                pickle.dump((train_dataset, val_dataset, test_dataset), f)
            return train_dataset, val_dataset, test_dataset


    def _read_classes(self, file):
        id2cls = json.load(open(file, 'r', encoding='utf-8'))
        cls2id = {}
        for i, x in enumerate(id2cls):
            cls2id[x] = i
        return id2cls, cls2id


