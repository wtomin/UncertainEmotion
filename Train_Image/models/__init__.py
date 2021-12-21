from .basenet import *
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
        bk = mobile_facenet(pretrained, args.cuda)
        bk.remove_output_layer()
        ninp = 512 # 512 is the feature dimension for each task, produced by GDC module

        temporal_models = {}
        for t in args.tasks:
            if t == 'FA':
                dim = 68*2
            elif t == 'EXPR' or t == 'AU':
                dim = len(PATH().Aff_wild2.categories[t])
            else:
                dim = len(PATH().Aff_wild2.categories[t])*20 # digitize_num
            
            if args.TModel == 'transformer':
                temporal_models[t] = TransformerModel(ninp, args.nhead, args.nhid, args.nlayers, dim, dropout)
            else:
                temporal_models[t] = RNNModel(args.TModel, ninp, args.nhid, args.nlayers, dim, dropout)

        STmodel = SpatialTemporalModel(bk, temporal_models, args.tasks)
        if args.cuda:
            STmodel.cuda()
        model = ModelWrapper(STmodel, args.name, args.tasks, args.checkpoints_dir,
            args.loggings_dir, args.load_epoch, args.batch_size, args.seq_len,
            args.image_size, args.lr, args.lr_policy, args.lr_decay_epochs, args.T_max,
            args.optimizer, args.weight_decay,
            args.gpu_ids, 
            args.AU_criterion, args.EXPR_criterion, args.VA_criterion, args.FA_criterion,
            args.lambda_AU, args.lambda_EXPR, args.lambda_VA, args.lambda_FA,
            is_train, args.cuda)
        return model