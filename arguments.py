import argparse
import os
"""
Detailed hyper-parameter configurations.
"""
class Param:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    def all_param(self, parser):
        ################################## load data ####################################

        parser.add_argument("--data_file", default="/content/drive/MyDrive/LDA_CRL/src/dataloaders/zingnews.json")


        parser.add_argument("--class_file", default="/content/drive/MyDrive/LDA_CRL/src/dataloaders/id2cls.json")

        parser.add_argument("--num_classes", default=8, type=int)

        parser.add_argument("--num_of_train", default=0.8, type=int)
        parser.add_argument("--num_of_val", default=0.1, type=int)
        parser.add_argument("--num_of_test", default=0.1, type=int)


        ##################################common parameters####################################
        parser.add_argument("--gpu", default=0, type=int)

        parser.add_argument("--dataname", default='zingnews', type=str, help="Choose dataset")


        parser.add_argument("--max_length", default=256, type=int)
        # parser.add_argument("--max_length", default="max_length")

        parser.add_argument("--device", default="cuda", type=str)

        ###############################   training ################################################

        parser.add_argument("--batch_size", default=16, type=int)

        parser.add_argument("--learning_rate", default=5e-6, type=float)
        
        parser.add_argument("--total_round", default=1, type=int)

        parser.add_argument("--encoder_output_size", default=768, type=int)

        parser.add_argument("--vocab_size", default=30522, type =int)

        parser.add_argument("--num_workers", default=0, type=int)

        # Temperature parameter in CL and CR
        parser.add_argument("--temp", default=0.1, type=float)

        # The projection head outputs dimensions
        parser.add_argument("--feat_dim", default=64, type=int)


        # epoch1
        parser.add_argument("--nepochs", default=10, type=int) 


        parser.add_argument("--seed", default=2021, type=int) 

        parser.add_argument("--max_grad_norm", default=10, type=float) 


        parser.add_argument("--optim", default='adam', type=str)

        # Memory size
        parser.add_argument("--num_protos", default=20, type=int)


        # dataset path
        parser.add_argument("--data_path", default='dat/', type=str)

        # bert-base-uncased weights path
        parser.add_argument("--bert_path", default="vinai/phobert-base", type=str)
        
        return parser