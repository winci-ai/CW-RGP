from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
import os
from cfg import get_cfg
from datasets import get_ds
from methods import get_method
import shutil

def get_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epochs if cfg.T0 is None else cfg.T0,
            T_mult=cfg.Tmult,
            eta_min=cfg.eta_min,
        )
    elif cfg.lr_step == "step":
        m = [cfg.epochs - a for a in cfg.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.drop_gamma)
    else:
        return None

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def lr_warmup(cfg, optimizer, step):
    if step < cfg.warmup_iters:
        lr_scale = (step + 1) / cfg.warmup_iters
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * lr_scale

def main():
    cfg = get_cfg()

    wandb.init(project=cfg.wandb, name=cfg.env_name, config=cfg)  
    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers)   
    model = get_method(cfg.method)(cfg)
    model.cuda().train()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(optimizer, cfg)

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
        for n_iter, (samples, _) in enumerate(tqdm(ds.train, position=1, ascii=True)):
            lr_warmup(cfg, optimizer, n_iter + epoch * iters)
            optimizer.zero_grad()
            loss = model(samples)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

            model.step(epoch / cfg.epochs)
            if cfg.lr_step == "cos" and cfg.warmup_cnt >= 500:
                scheduler.step(epoch + n_iter / iters)

        if cfg.lr_step == "step":
            scheduler.step()
    
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

        wandb.log({"loss": np.mean(loss_ep), "ep": epoch})
        

if __name__ == "__main__":
    main()