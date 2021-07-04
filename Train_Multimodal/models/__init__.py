from PATH import PATH
from .TemporalModel import *
from .SpatialTemporalModel import SpatialTemporalModel
from .models import ModelWrapper
from utils.misc import get_MarbleNet_config, Identity, VAD_MarbleNet, mobile_facenet
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
            audio_bk = VAD_MarbleNet.from_pretrained(model_name="vad_marblenet")
        else:
            config = get_MarbleNet_config()
            audio_bk = VAD_MarbleNet(cfg=config.model)
        audio_bk.decoder = Identity()
        video_bk = mobile_facenet(pretrained, args.cuda)
        video_bk.remove_output_layer()
        ninp = 128 + 512 

        temporal_models = {}
        for t in args.tasks:
            if t == 'FA':
                dim = 68*2
            elif t == 'VAD':
                dim = 2
            elif t == 'EXPR' or t == 'AU':
                dim = len(PATH().Aff_wild2.categories[t])
            else:
                dim = len(PATH().Aff_wild2.categories[t])*20 # digitize_num
            
            if args.TModel == 'transformer':
                temporal_models[t] = TransformerModel(ninp, args.nhead, args.nhid, args.nlayers, dim, dropout)
            else:
                temporal_models[t] = RNNModel(args.TModel, ninp, args.nhid, args.nlayers, dim, dropout)
        
        STmodel = SpatialTemporalModel(video_bk, audio_bk, temporal_models, args.tasks)
        if args.cuda:
            STmodel.cuda()
        model = ModelWrapper(STmodel, args.name, args.tasks, args.checkpoints_dir,
            args.loggings_dir, args.load_epoch, args.batch_size, args.seq_len, args.image_size,
            args.window_size, args.sr,
            args.lr, args.lr_policy, args.lr_decay_epochs, args.T_max,
            args.optimizer, args.weight_decay,
            args.gpu_ids, 
            args.AU_criterion, args.EXPR_criterion, args.VA_criterion, args.FA_criterion, args.VAD_criterion,
            args.lambda_AU, args.lambda_EXPR, args.lambda_VA, args.lambda_FA, args.lambda_VAD, 
            is_train, args.cuda)
        return model