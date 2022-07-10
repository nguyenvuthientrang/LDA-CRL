import torch
from arguments import Param
from utils import setup_seed
from trainer.crl import Trainer
def run(args):
    setup_seed(args.seed)
    print("hyper-parameter configurations:")
    print(str(args.__dict__))
    
    trainer = Trainer(args)
    trainer.train(args)


if __name__ == '__main__':
    param = Param() # There are detailed hyper-parameter configurations.
    args = param.args
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)
    args.n_gpu = torch.cuda.device_count()
    args.task_name = args.dataname
    run(args)