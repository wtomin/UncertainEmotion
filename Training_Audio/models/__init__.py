from PATH import PATH
from .TemporalModel import *
from .SpatialTemporalModel import SpatialTemporalModel
from .models import ModelWrapper
from utils.misc import get_MarbleNet_config, Identity, VAD_MarbleNet
from nemo.collections.asr.modules import ConvASRDecoderClassification
from copy import copy
class ModelsFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(args, 
        is_train, 
        dropout = 0.5,
        pretrained = True):
        if pretrained:
            bk = VAD_MarbleNet.from_pretrained(model_name="vad_marblenet")
        else:
            config = get_MarbleNet_config()
            bk = VAD_MarbleNet(cfg=config.model)
        bk.decoder = Identity()
        # bk.loss = Identity()
        # bk._accuracy = Identity()
        ninp = 128

        temporal_models = {}
        for t in args.tasks:
            if t == 'EXPR':
                dim = len(PATH().Aff_wild2.categories[t])
            elif t == 'VA':
                dim = len(PATH().Aff_wild2.categories[t])*20 # digitize_num
            elif t == 'VAD':
                dim = 2
            temporal_models[t] = ConvASRDecoderClassification(ninp, dim, return_logits=True, pooling_type='avg')
        STmodel = SpatialTemporalModel(bk, temporal_models, args.tasks, dropout)
        if args.cuda:
            STmodel.cuda()
        model = ModelWrapper(STmodel, args.name, args.tasks, args.checkpoints_dir,
            args.loggings_dir, args.load_epoch, args.batch_size, args.time_length,
            args.lr, args.lr_policy, args.lr_decay_epochs, args.T_max,
            args.optimizer, args.weight_decay,
            args.gpu_ids, 
            args.EXPR_criterion, args.VA_criterion, args.VAD_criterion,
            args.lambda_EXPR, args.lambda_VA, args.lambda_VAD, args.sr,
            is_train, args.cuda)
        return model