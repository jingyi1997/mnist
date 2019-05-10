from __future__ import print_function
import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import distributed
from torch.autograd import Variable
from tensorboardX import SummaryWriter
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb

from memcached_dataset import McDataset
import torchvision.transforms as transforms
from utils import *
import model
parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--base_lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='')
parser.add_argument('--ckpt', default='experiment/softmax', type=str, help='')
parser.add_argument('--data_dir', default='/mnt/lustre/xujingyi/sphereface/preprocess/result/CASIA-WebFace-112X96', type=str)
parser.add_argument('--val_data_dir', default='data/lfw-112X96', type=str)
parser.add_argument('--val_data_list', default='meta/0.9_0.9_0.9_test.txt', type=str)
parser.add_argument('--data_list', default='/mnt/lustre/xujingyi/sphereface/preprocess/result/webface.txt', type=str)
parser.add_argument('--loss_type', default='softmax', type=str)
parser.add_argument('--transforms', default='softmax', type=str)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--save_freq', default=1000, type=str)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--vis_freq', default=10, type=int)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--classnum', default=10574, type=int)
parser.add_argument('--distributed', default=False, type=str)
parser.add_argument('--lr_steps', nargs='+', type=int)
parser.add_argument('--evaluate', default=None, type=str)
parser.add_argument('--lr_mults', default=0.1, type=float)
parser.add_argument('--gamma', default=0.12, type=float)
parser.add_argument('--sample_feat', default=None, type=str)
parser.add_argument('--var_weight', default=0, type=float)
parser.add_argument('--power', default=1, type=float)
parser.add_argument('--margin', default=0.5, type=float)
parser.add_argument('--LambdaMax', default=1000, type=float)
parser.add_argument('--radius', default=None, type=float)
parser.add_argument('--pretrained', default=None, type=str)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()



def main():

    global  writer, best_prec1
      

    if not os.path.exists('{}/checkpoints'.format(args.ckpt)):
         os.makedirs('{}/checkpoints'.format(args.ckpt))
    if not os.path.exists('{}/images'.format(args.ckpt)):
         os.makedirs('{}/images'.format(args.ckpt))
    if not os.path.exists('{}/logs'.format(args.ckpt)):
         os.makedirs('{}/logs'.format(args.ckpt))
    if not os.path.exists('{}/plots'.format(args.ckpt)):
         os.makedirs('{}/plots'.format(args.ckpt))
    if os.path.exists('{}/runs'.format(args.ckpt)):
         shutil.rmtree('{}/runs'.format(args.ckpt))
    os.makedirs('{}/runs'.format(args.ckpt))


    logger = create_logger('global_logger', '{}/logs/{}.txt'.format(args.ckpt,time.time()))
    logger.info('{}'.format(args))
    writer = SummaryWriter('{}/runs'.format(args.ckpt))
    
    #net = nn.parallel.distributed.DistributedDataParallel(net)
    
    # build dataset
    data_dir = args.data_dir
    data_list = args.data_list
    
    val_data_dir = args.val_data_dir
    val_data_list = args.val_data_list

    
    train_dataset = McDataset(data_dir, data_list)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        num_workers=1, pin_memory=True, collate_fn=fast_collate)

    val_dataset = McDataset(val_data_dir, val_data_list)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.bs, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=fast_collate)

    # create model
    
    print("=> creating model '{}'".format(args.net))
    net = model.Net(classnum=args.classnum, feature_dim=2, head=args.loss_type, radius=args.radius, sample_feat=args.sample_feat) 
    net = net.cuda()
    print(net)
    
    # build optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
 
    if args.loss_type == 'a-softmax':
        criterion = model.AngleLoss(LambdaMax=args.LambdaMax, gamma=args.gamma, power=args.power).cuda()
    if args.loss_type == 'softmax' or args.loss_type == 'gaussian':
        criterion = torch.nn.CrossEntropyLoss()
  
    start_epoch = 0
    best_prec1 = 0
    # optionally resume from a pretrained model
    
    if args.evaluate:
        model_path = os.path.join(args.ckpt, 'checkpoints', args.evaluate)
        checkpoint = torch.load(model_path)
        epoch  = int(checkpoint['epoch'])
        net.load_state_dict(checkpoint['state_dict'])
        val_feat, val_target, prec1 = validate(net, val_loader, criterion, epoch)
        print('Prec1: {}'.format(prec1))
        return  


    if args.resume:
        model_path = os.path.join(args.ckpt, 'checkpoints', args.resume)
        checkpoint = torch.load(model_path)
        start_epoch = int(checkpoint['epoch'])
        best_prec1 = float(checkpoint['best_prec1'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.pretrained:
        start_epoch = 0

    ## start training 
    net.train()
    freq = args.print_freq
    end = time.time()
    
    for epoch in range(start_epoch,args.epochs):
        train_feat, train_target, train_prec1 = train(net, epoch, train_loader, args, criterion, optimizer)
        val_feat, val_target, prec1 = validate(net, val_loader, criterion, epoch)
        #pdb.set_trace()
        if (epoch+1) % args.vis_freq == 0:
           visualize(train_feat, train_target, val_feat, val_target, epoch, args, train_prec1, prec1)
        is_best = prec1>best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
               'epoch': epoch+1,
               'state_dict': net.state_dict(),
               'optimizer': optimizer.state_dict(),
               'best_prec1': best_prec1
        },args, is_best)
    print('Best Prec1: {}'.format(best_prec1))
    writer.close()

def validate(net, val_loader,criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    feature_all = []
    target_all = []
    logits_all = []
    norm_all = torch.Tensor()
    num_all = torch.Tensor()

    # switch to evaluate mode
    net.eval()
    net.feature = True
    end = time.time()

    prefetcher = DataPrefetcher(val_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            feature, logits, var = net(input)
            if args.loss_type == 'softmax' or args.loss_type == 'gaussian':
              cos_theta = logits
              loss  = criterion(cos_theta, target)
              #pdb.set_trace()
              prec1 = accuracy(cos_theta.data, target, topk=(1,))
            if args.loss_type == 'a-softmax':
              cos_theta, phi_theta = logits
              loss = criterion((cos_theta, phi_theta), target)
              prec1 = accuracy(cos_theta.data, target, topk=(1,))

        feature_all.append(feature)
        target_all.append(target)
        # measure accuracy and record loss
        reduced_loss = loss.data
        
        losses.update(to_python_float(reduced_loss))
        top1.update(to_python_float(prec1[0]))
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger = logging.getLogger('global_logger')
        niters = epoch*len(val_loader) + i
        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader),
                   batch_time=batch_time, loss=losses,
                   top1=top1))
        #writer.add_scalar('test_prec1', top1.val, niters)
        input, target = prefetcher.next()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    feature_all = torch.cat(feature_all, 0)
    target_all = torch.cat(target_all, 0)
    return feature_all.data.cpu().numpy(), target_all.data.cpu().numpy(), top1.avg


def train(net, epoch,train_loader, args, criterion, optimizer):
      freq = args.print_freq
      losses = AverageMeter(freq)
      data_time = AverageMeter(freq)
      batch_time = AverageMeter(freq)
      top1 = AverageMeter(freq)

      net.train()
      net.feature = False
      end = time.time()
 
      prefetcher = DataPrefetcher(train_loader)
      input, target = prefetcher.next()
      i=-1
      feature_all = []
      target_all = []
      while input is not None:
          i += 1
          lr = adjust_learning_rate(optimizer, epoch, args)
          data_time.update(time.time() - end)
          feature, logits, var = net(input)
          feature_all.append(feature)
          target_all.append(target)

          if args.loss_type == 'softmax' or args.loss_type == 'gaussian':
              cos_theta = logits
              softmax_loss =  criterion(cos_theta, target)
              prec1 = accuracy(cos_theta.data, target, topk=(1,))
          if args.loss_type == 'a-softmax':
              cos_theta, phi_theta = logits
              softmax_loss, lamda = criterion((cos_theta, phi_theta), target)
              prec1 = accuracy(cos_theta.data, target, topk=(1,))

          var_loss = torch.mean(torch.pow(1-var,2))
          if args.var_weight == 0:
              loss = softmax_loss
          else:
              loss = softmax_loss + args.var_weight * var_loss

          reduced_loss = loss.data
          losses.update(to_python_float(reduced_loss))
          top1.update(to_python_float(prec1[0]))
            
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          torch.cuda.synchronize()

          batch_time.update(time.time() - end)

          end = time.time()
          input, target = prefetcher.next()

          if i % args.print_freq == 0:
              niters = epoch * len(train_loader) + i
              logger = logging.getLogger('global_logger')
              loss_info = 'Epoch: [{0}]/[{1}/{2}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Softmax Loss {softmax_loss: .4f}\t' \
                      'Var Loss {var_loss: .4f}\t' \
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                      'LR {lr:.4f}'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time, loss=losses, softmax_loss=softmax_loss.data.item(),
                       var_loss=var_loss.data.item(), top1=top1, lr=lr)
              writer.add_scalar('softmax_loss', softmax_loss.data.item(), niters)
              writer.add_scalar('var_loss', var_loss.data.item(), niters)
              writer.add_scalar('train_prec1', top1.val, niters)
              if args.loss_type == 'a-softmax':
                loss_info = loss_info + '\tLamda {lamda: .3f}'.format(lamda=lamda)

              logger.info(loss_info)
      feature_all = torch.cat(feature_all, 0)
      target_all = torch.cat(target_all, 0)
      return feature_all.data.cpu().numpy(), target_all.data.cpu().numpy(), top1.avg

      

if __name__ == '__main__':
    main()
