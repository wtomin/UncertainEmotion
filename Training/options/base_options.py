import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--load_epoch', type=int, default=-1, 
            help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--AU_criterion', type=str, default = 'bce')
        self._parser.add_argument('--EXPR_criterion', type=str, default = 'ce')
        self._parser.add_argument('--VA_criterion', type=str, default = 'ccc')
        self._parser.add_argument('--FA_criterion', type=str, default= 'l1_loss')
        self._parser.add_argument('--lambda_AU', type=float, default=1)
        self._parser.add_argument('--lambda_EXPR', type=float, default=1)
        self._parser.add_argument('--lambda_VA', type=float, default=1)
        self._parser.add_argument('--lambda_FA', type=float, default=1)
        ########## Data and tasks #########
        self._parser.add_argument('--dataset_names', type=str, default = ['Mixed_EXPR','Mixed_AU','Mixed_VA'],nargs="+")
        self._parser.add_argument('--tasks', type=str, default = ['EXPR','AU','VA'],nargs="+")
        self._parser.add_argument('--seq_len', type=int, default= 30, help='length of input seq ')
        self._parser.add_argument('--fps', type=int, default=30, help=
            "Changing the fps to some integer smaller than 30 can change the sampling rate")
        self._parser.add_argument('--batch_size', type=int, default= 2, help='input batch size per task')
        self._parser.add_argument('--image_size', type=int, default= 112, help='input image size') 

        ########### Ablation study: w/o auxillary task; Transformer or RNN #########
        self._parser.add_argument('--TModel', type=str, default='GRU',
                            help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
        self._parser.add_argument('--auxillary', action='store_true', help=
            "Whether to train face alignment as an auxillary task.")
        ########## temporal model definition #########
        self._parser.add_argument('--nhead', type=int, default=2,
                            help='the number of heads in the encoder/decoder of the transformer model')
        self._parser.add_argument('--nhid', type=int, default=128,
                            help='number of hidden units per layer')
        self._parser.add_argument('--nlayers', type=int, default=1,
                            help='number of layers in the temporal model')
        self._parser.add_argument('--optimizer', type=str, default='Adam')
        self._parser.add_argument('--gpu_ids', type=str, default='0', nargs='+',
            help='gpu ids: e.g. 0 , 0 1 2. use -1 for CPU')
        self._parser.add_argument('--cuda', action='store_true', help="Whether to use GPU")
        self._parser.add_argument('--print_freq_s', type=int, default= 10, help='print the training loss after every # seconds')
        self._parser.add_argument('--save_freq_s', type=int, default= 10,
            help= 'save the training losses to the summary writer every # seconds.')
        self._parser.add_argument('--n_threads_train', default=8, type=int, help='# threads for loading data')
        self._parser.add_argument('--n_threads_test', default=2, type=int, help='# threads for loading data')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--loggings_dir', type=str, default='./loggings', help='loggings are saved here')
        self._parser.add_argument('--lr', type=float, default=1e-3, 
            help= "The initial learning rate")
        self._parser.add_argument('--lr_policy', type=str, default='step', choices=['step', 'cosine'])
        self._parser.add_argument('--lr_decay_epochs', type=int, default=3, help='reduce the lr to 0.1*lr for every # epochs')
        self._parser.add_argument('--T_max', type=int, default=20000, help='the period for the cosine annealing (# iterations)')
        self._parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
        self._parser.add_argument('--nepochs', type=int, default=10)

        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or test
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        if self.is_train and not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        else:
            assert os.path.exists(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
