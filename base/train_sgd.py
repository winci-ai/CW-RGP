from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from cfg import get_cfg
from datasets import get_ds
from methods import get_method
import shutil
import math

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    cur_lr = cfg.lr * 0.5 * (1. + math.cos(math.pi * epoch / cfg.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = cfg.lr
        else:
            param_group['lr'] = cur_lr

def main():
    cfg = get_cfg()

    wandb.init(project=cfg.wandb, name=cfg.env_name, config=cfg)  
    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers)   
    model = get_method(cfg.method)(cfg)
    model.cuda().train()

    optim_params = model.parameters()
    if cfg.method == 'simsiam':
        other_param = []
        for param in model.model.parameters():
            other_param.append(param)
        for param in model.head.parameters():
            other_param.append(param)

        optim_params = [{'params': other_param, 'fix_lr': False},
                        {'params': model.pred.parameters(), 'fix_lr': True}]

    optimizer = optimizer = optim.SGD(optim_params, cfg.lr, momentum=0.9 , 
                                        weight_decay=cfg.weight_decay)
    eval_every = cfg.eval_every
    cudnn.benchmark = True

    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu is None:
                checkpoint = torch.load(cfg.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    for epoch in trange(cfg.start_epoch, cfg.epochs):
        loss_ep = []
        iters = len(ds.train)
        adjust_learning_rate(optimizer, epoch, cfg)
        for n_iter, (samples, _) in enumerate(tqdm(ds.train, position=1, ascii=True)):
            optimizer.zero_grad()
            loss = model(samples)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

            model.step(epoch / cfg.epochs)
    
        if len(cfg.drop) and epoch == (cfg.epochs - cfg.drop[0]):
            eval_every = cfg.eval_every_drop

        if (epoch + 1) % eval_every == 0:
            acc_knn, acc = model.get_acc(ds.clf, ds.test)
            print("acc:",acc[1], "acc_5:",acc[5], "acc_knn:",acc_knn)
            wandb.log({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}, commit=False)
    
        if (epoch + 1) % 500 == 0:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': cfg.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='checkpoint_{}_{:04d}.pth.tar'.format(cfg.env_name, epoch))

        wandb.log({"loss": np.mean(loss_ep), "ep": epoch, "learning_rate_1": optimizer.param_groups[0]['lr'],
                            "learning_rate_2": optimizer.param_groups[1]['lr']})
        

if __name__ == "__main__":
    main()