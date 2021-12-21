import os
import torch
def _set_and_check_load_epoch(models_dir, load_epoch):
    if os.path.exists(models_dir):
        if load_epoch == -1:
            load_epoch = 0
            for file in os.listdir(models_dir):
                if file.startswith("epoch_"):
                    load_epoch = max(load_epoch, int(file.split('_')[1]))
            load_epoch = load_epoch
        else:
            found = False
            for file in os.listdir(models_dir):
                if file.startswith("epoch_"):
                    found = int(file.split('_')[1]) == load_epoch
                    if found: break
            assert found, 'Model for epoch %i not found' % load_epoch
    else:
        assert load_epoch < 1, 'Model for epoch %i not found' % load_epoch
        load_epoch = 0
    return load_epoch

def _get_set_gpus(gpu_ids):
    # get gpu ids
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    return gpu_ids

def _print(args):
    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

def _save(save_dir, is_train, args):
    expr_dir = save_dir
    if is_train:
        os.makedirs(expr_dir)
    else:
        assert os.path.exists(expr_dir)
    file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if is_train else 'test'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(vars(args).items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

def prepare_arguments(args, is_train):
    exp_dir = os.path.join(args.checkpoints_dir, args.name)
    args.load_epoch = _set_and_check_load_epoch(exp_dir, args.load_epoch)
    if args.cuda:
        args.gpu_ids = _get_set_gpus(args.gpu_ids)
    _print(args)
    _save(exp_dir, is_train=is_train , args= args)