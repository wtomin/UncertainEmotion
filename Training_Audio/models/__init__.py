from PATH import PATH
from .TemporalModel import *
from .SpatialTemporalModel import SpatialTemporalModel
from .models import ModelWrapper
from utils.misc import mobile_facenet
class ModelsFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(args, 
        is_train, 
        dropout = 0.5,
        pretrained = True):
        bk = mobile_facenet(pretrained = False) # randomly intialized
        bk.remove_output_layer()
        ninp = 512

        temporal_models = {}
        for t in args.tasks:
            if t == 'EXPR':
                dim = len(PATH().Aff_wild2.categories[t])
            else:
                dim = len(PATH().Aff_wild2.categories[t])*20 # digitize_num
            
            if args.TModel.lower() == 'transformer':
                temporal_models[t] = TransformerModel(ninp, args.nhead, args.nhid, args.nlayers, dim, dropout)
            else:
                temporal_models[t] = RNNModel(args.TModel, ninp, args.nhid, args.nlayers, dim, dropout)

        STmodel = SpatialTemporalModel(bk, temporal_models, args.tasks)
        if args.cuda:
            STmodel.cuda()
        model = ModelWrapper(STmodel, args.name, args.tasks, args.checkpoints_dir,
            args.loggings_dir, args.load_epoch, args.batch_size, args.seq_len,
            args.lr, args.lr_policy, args.lr_decay_epochs, args.T_max,
            args.optimizer, args.weight_decay,
            args.gpu_ids, 
            args.EXPR_criterion, args.VA_criterion, 
            args.lambda_EXPR, args.lambda_VA, args.sr,
            is_train, args.cuda)
        return model