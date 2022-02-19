import pytorch_lightning as pl
from utils.misc import mobile_facenet
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.validation import CCCLoss, CCC_score_Torch
import numpy as np
import os
from data.dataset import DataModule
from utils.transforms import train_transforms, test_transforms
from utils.validation import AU_metric, EXPR_metric, VA_metric
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from PATH import PATH
import torchmetrics
PRESET_VARS = PATH()

class MarginCosineProduct(nn.Module):
    def __init__(self, in_features, out_features, s = 3.65, m=0., use_bias = True):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  
        self.m = m  
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        stdv = 1./np.sqrt(self.weight.size(1))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = None
    def forward(self, input, label = None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is not None:
            if len(label.size()) == 1:
                onehot = torch.zeros_like(cosine)
                onehot.scatter_(1, label.view(-1, 1), 1.0).to(label.device)
            else:
                onehot = label
            if self.bias is not None:
                output = self.s * (cosine - onehot * self.m) + self.bias
            else:
                output = self.s * (cosine - onehot * self.m) 
        else:
            # margin is not used when label is None
            output = cosine
        return output

def cosine_theta2cube_magnitude(cosine_theta):
    # the input x is theta, output y is the magnitude in the cube
    # x: [0, pi/4], y = 1/|cos(theta)|
    # x: [pi/4, pi/2] y = 1/cos(pi/2 - theta) = 1/|sin(theta)|
    # x: [pi/2, 3pi/4] , y = 1/cos(theta - pi/2) = 1/|sin(theta)|
    # x: [3pi/4, pi] , y = 1/cos(pi - theta) = 1/|cos(theta)|
    # y is always positive
    sin_theta_pos = torch.sqrt( 1- torch.pow(cosine_theta, 2))
    theta = torch.arccos(cosine_theta)
    mask_cosine = torch.abs(cosine_theta)>=np.sqrt(2)/2
    mask_sine = ~mask_cosine
    output = torch.zeros_like(cosine_theta)
    output[mask_cosine] = 1/torch.abs(cosine_theta[mask_cosine])
    output[mask_sine] = 1/sin_theta_pos[mask_sine]
    output = output.to(cosine_theta.device)
    return output

def plot_cube_magnitude():
    theta = np.linspace(0, np.pi, 50)
    theta = torch.FloatTensor(theta)
    cosine_theta = torch.cos(theta)
    cube_magnitude = cosine_theta2cube_magnitude(cosine_theta)
    from matplotlib import pyplot as plt
    plt.plot(theta.numpy(), cube_magnitude.numpy())
    plt.show()

def cube2sphere(va):
    # from [-1, 1] cube to a unit sphere
    x, y = va[:, 1], va[:, 0]
    L2_square = x**2 + y**2
    L2 = torch.sqrt(L2_square)
    cosine_theta = x/(L2+ 1e-9)
    normalization_magnitude = cosine_theta2cube_magnitude(cosine_theta)
    x = x / normalization_magnitude
    y = y / normalization_magnitude
    return torch.stack([y, x], dim=-1)

def sphere2cube(va):
    # from a unit sphere to [-1, 1] cube
    x, y = va[:, 1], va[:, 0]
    L2_square = x**2 + y**2 
    L2 = torch.sqrt(L2_square)
    cosine_theta = x/(L2+ 1e-9)
    normalization_magnitude = cosine_theta2cube_magnitude(cosine_theta)
    x = x * normalization_magnitude
    y = y * normalization_magnitude
    return torch.stack([y, x], dim=-1)

class EmotionNet(pl.LightningModule):
    def __init__(self, tasks, lr=1e-3, T_max = 1e4, wd=0.):
        super().__init__()
        backbone = mobile_facenet(pretrained=True)
        backbone.remove_output_layer()
        self.backbone = backbone
        #self.fc = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024))
        emotion_layers = []
        for t in tasks:
            dim = len(PATH().Aff_wild2.categories[t])
            if t == 'AU':
                emotion_layer = nn.ModuleList(
                [nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 64), nn.BatchNorm1d(64)),
                MarginCosineProduct(64, dim, s = 4, m=0, use_bias=False)]
                )
            elif t == 'EXPR':
                emotion_layer = nn.ModuleList(
                [nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 64), nn.BatchNorm1d(64)),
                MarginCosineProduct(64, dim, s = 3.65, m=0, use_bias=False)]
                )
            elif t == 'VA':
                emotion_layer = nn.ModuleList(
                [nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 64), nn.BatchNorm1d(64)),
                MarginCosineProduct(64, dim, s = 3.65, m=0, use_bias=False)]
                )
            emotion_layers.append(emotion_layer)
        self.emotion_layers = nn.ModuleList(emotion_layers)
        self.tasks = tasks
        self.lr = lr
        self.wd = wd
        self.T_max = T_max
        self.AU_pos_weight = torch.tensor([23/3, 47/3, 21/4, 37/13, 3/2, 13/7, 3, 97/3, 97/3, 97/3, 37/63, 23/2])
        self.EXPR_weight = torch.tensor([2.5, 25, 40, 33, 4, 5.88, 12.5])
 
    def verify_metrics_integrity(self):
        #conclusion
        #1. EXPR metric is the same
        # 2. AU metric, numpy is wrong, torch is correct, and higher
        # 3. VA metric, probably epsilon errors
        for task in self.tasks:
            if task =='AU':
                preds = np.array([[0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 1, 0, 1, 1, 1]]).astype(np.int64)
                targets = np.array([[1, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 1, 1],
                    [1, 1, 1, 0, 1, 0, 1]]).astype(np.int64)
                from utils.validation import AU_metric
                print("{} Numpy metrics:{}".format(task, AU_metric(preds, targets)))
            elif task == 'EXPR':
                preds = np.array([1, 5, 6, 0, 6, 2, 3 , 4, 4, 1, 5, 6, 0, 6, 5, 3 , 1, 5]).astype(np.int64)
                targets = np.array([0, 5, 6, 0, 6, 2, 3 , 0, 4, 0, 5, 6, 0, 6, 0, 3 , 0, 4]).astype(np.int64)
                from utils.validation import EXPR_metric
                print("{} Numpy metrics:{}".format(task, EXPR_metric(preds, targets)))
            else:
                preds = np.array([[0.23, 0.523, 0.73, -0.24, -0.3, 0.3, 0.87],
                    [0.5, 0.42, 0.0, -0.2, -0.3, 0.45, 0.2]]).astype(np.float32)
                targets = np.array([[0.01, 0.23, 0.654, -0.1, -0.9, 0.1, 0.6],
                    [0.1, 0.32, 0.1, -0.756, -0.4, 0.4, 0.4]]).astype(np.float32)
                from utils.validation import VA_metric
                print("{} Numpy metrics:{}".format(task, VA_metric(preds, targets)))
            res = self.compute_metric(task, torch.tensor(preds), torch.tensor(targets))
            print("{} Torch metrics:{}".format(task, res))
    def forward(self, x, y = {}):
        x = self.backbone(x)
        output = {}
        for i, t in enumerate(self.tasks):
            spherical_feature = self.emotion_layers[i][0](x)
            if t != 'VA' and t in y.keys():
                out_emo = self.emotion_layers[i][1](spherical_feature, y[t])
            else:
                out_emo = self.emotion_layers[i][1](spherical_feature)
            output[t] = out_emo
        return output
    def compute_loss(self, task, y_hat, y):
        if task == 'AU':
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight = self.AU_pos_weight.to(y.device))
        elif task == 'EXPR':
            return F.cross_entropy(y_hat, y, weight = self.EXPR_weight.to(y.device))
        else:
            y = cube2sphere(y)
            return VA_LOSS(y_hat[..., 0], y[..., 0]) + VA_LOSS(y_hat[..., 1], y[..., 1])
    def compute_metric(self, task, estimates, labels):
        if task=='AU':
            f1_metric = torchmetrics.F1(num_classes = 2, average='macro', mdmc_average='samplewise')
            acc_metric = torchmetrics.Accuracy()
            f1 = f1_metric(estimates, labels).item()
            acc = acc_metric(estimates, labels).item()
            return [f1, acc], 0.5*f1+0.5*acc
        elif task == 'EXPR':
            f1_metric = torchmetrics.F1(num_classes=7, average='macro') # EXPR f1 score macro
            acc_metric = torchmetrics.Accuracy()
            f1 = f1_metric(estimates, labels).item()
            acc = acc_metric(estimates, labels).item()
            return [f1, acc], 0.67*f1+0.33*acc
        else:
            estimates = sphere2cube(estimates)
            v_ccc = CCC_score_Torch(estimates[:,0], labels[:, 0]).item()
            a_ccc = CCC_score_Torch(estimates[:,1], labels[:, 1]).item()
            return [v_ccc, a_ccc], v_ccc+a_ccc
    def compute_metric_numpy(self, task, estimates, labels):
        if task=='AU':
            res = AU_metric(estimates, labels)
        elif task=='EXPR':
            res = EXPR_metric(estimates, labels)
        else:
            estimates = sphere2cube(torch.tensor(estimates)).numpy()
            res = VA_metric(estimates, labels)
        return res
    def compute_estimate(self, task, y_hat):
        if task =='AU':
            y_hat = (torch.sigmoid(y_hat).cpu()>0.5).type(torch.LongTensor)
        elif task =='EXPR':
            y_hat = F.softmax(y_hat, dim=-1).cpu().argmax(-1).type(torch.LongTensor)
        else:
            y_hat = y_hat.cpu()
        return y_hat

    def training_step(self, batch, batch_idx):
        total_loss = 0
        for t in self.tasks:
            x, y, _, _ = batch[t]
            y_hat = self(x, y = {t: y})
            loss = self.compute_loss(t, y_hat[t], y)
            self.log('train_loss/{}'.format(t), loss, on_step=True, on_epoch=True, 
                prog_bar=True, logger=True)
            total_loss+=loss
        return total_loss
    def validation_step(self, batch, batch_idx, dataloader_idx):
        # multiple validation datasets
        task = self.tasks[dataloader_idx]
        x, y, _, _ = batch
        y_hat = self(x, y = {task: y})
        preds = {}
        for t in self.tasks: # multitask prediction
            estimate = self.compute_estimate(t, y_hat[t])
            preds[t] = estimate
        return preds, y
    def validation_epoch_end(self, validation_step_outputs):
        #check the validation step outputs
        num_dataloaders = len(validation_step_outputs)
        num_batches = len(validation_step_outputs[0]) # whether the number of batches are the same of all dataloaders?
        print("id [{}]: {} id [{}]: {} id [{}]: {}".format(
            1, len(validation_step_outputs[0]),
            2, len(validation_step_outputs[1]),
            3, len(validation_step_outputs[2])))
        total_metric = {'torch': 0, 'numpy':0 }
        for i_task in range(num_dataloaders):
            task = self.tasks[i_task]
            outputs = validation_step_outputs[i_task]
            preds, labels = [], []
            for p, y in outputs:
                preds.append(p[task])
                labels.append(y)
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0).cpu()
            if task in ['EXPR', 'AU']:
                labels = labels.type(torch.LongTensor)
            for metric_name in ['torch','numpy']:
                if metric_name =='torch':
                    metrics, metric = self.compute_metric(task, preds, labels)
                else:
                    metrics, metric = self.compute_metric_numpy(task, preds.numpy(), labels.numpy())
            
                self.log('val_metric_{}/{}_{}'.format(metric_name, task, '1'), metrics[0], on_epoch=True, logger=True)
                self.log('val_metric_{}/{}_{}'.format(metric_name,task, '2'),metrics[1], on_epoch=True,  logger=True)
                self.log('val_metric_{}/{}'.format(metric_name,task),metric, on_epoch=True, logger=True)
                total_metric[metric_name]+=metric
        
        self.log('val_metric_{}'.format("total_{}".format('torch')),total_metric['torch'], on_epoch=True, 
                        logger=True)
        self.log('val_metric_{}'.format("total_{}".format('numpy')),total_metric['numpy'], on_epoch=True, 
                        logger=True)
    
    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.parameters(),
                lr=self.lr,
                weight_decay = self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                self.T_max)

        return {
        'optimizer': optimizer,
        'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': "lr"}
        }

class MSE_alpha_Cov(nn.Module):
    def __init__(self, alpha = 0.1):
        super().__init__()
        self.alpha = alpha
        assert self.alpha >0, "positive alpha"

    def forward(self, x, y):
        mse_loss = F.mse_loss(x, y)
        cov =  (x* y).mean(dim=0)
        return mse_loss - self.alpha * cov
    

if __name__ == '__main__':
    #plot_cube_magnitude()
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--ckp-save-dir', type=str, default='checkpoints')
    parser.add_argument('--exp-name', type=str, default='experiment_1')
    parser.add_argument('--find-best-lr', action="store_true")
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--va_loss', type=str, default = 'ccc')
    parser.add_argument('--resume_ckp', type=str, default=None)
    args = parser.parse_args()
    tasks = ['AU', 'EXPR', 'VA']
    global VA_LOSS
    if args.va_loss == 'ccc':
        VA_LOSS = CCCLoss(digitize_num=1)
    elif args.va_loss == 'mse':
        VA_LOSS = nn.MSELoss()
    elif args.va_loss == 'mae':
        VA_LOSS = nn.L1Loss()
    elif args.va_loss == 'mse_cov':
        VA_LOSS = MSE_alpha_Cov(alpha = 0.1)

    model = EmotionNet(tasks, lr = args.lr, wd=args.wd)
    # model.verify_metrics_integrity()
    dm = DataModule(tasks,
        transform_train = train_transforms(112), transform_test = test_transforms(112), 
        num_workers_train = 8, num_workers_test = 8, batch_size = 24,
        downsamples = [4, 2, 4])
    ckp_dir = os.path.join(args.ckp_save_dir, args.exp_name)
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    early_stopper = EarlyStopping(monitor='val_metric_total_numpy',
        min_delta=0., patience=10, verbose=True, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckp_callback = ModelCheckpoint(monitor='val_metric_total_numpy', mode='max',
        dirpath = ckp_dir,
        filename = 'mobile_facenet-{epoch:02d}-{val_metric_total_numpy:.2f}',
        save_top_k = 1,
        save_last = True)
    tb_logger = pl_loggers.TensorBoardLogger(ckp_dir)
    trainer = Trainer(gpus=1, benchmark=True,
        default_root_dir = ckp_dir, logger = tb_logger, log_every_n_steps=100, 
        max_steps = 10e5, 
        # limit_train_batches = 0.01, 
        # limit_val_batches= 0.01, 
        callbacks =[early_stopper, lr_monitor, ckp_callback],
        resume_from_checkpoint = args.resume_ckp)

    if args.find_best_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule = dm, 
            min_lr = 1e-5, max_lr = 1e-1, )
        fig = lr_finder.plot(suggest=True)
        fig.show()
    else:
        trainer.fit(model, datamodule=dm)
