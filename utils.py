import torch
import time
import os
import torch.distributed as dist
import multiprocessing as mp
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.io import savemat
import shutil


def visualize(train_feat, train_label, test_feat, test_label, epoch, args, train_acc=None, test_acc=None):
    # plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    feats = [train_feat, test_feat]
    labels = [train_label, test_label]
    if train_acc is not None and test_acc is not None:
        accs = [train_acc, test_acc]
        titles = ['train_acc: ', 'test_acc: ']
    else:
        accs = None
        titles = None
    plt.figure(figsize=(9, 4))
    for i in range(2):
        feat = feats[i]
        label = labels[i]
        plt.subplot(1, 2, i+1)
        for j in range(args.classnum):
            plt.plot(feat[label == j, 0], feat[label == j, 1], '.', c=c[j])
        if accs is not None:
            plt.title(titles[i]+'%.3f%%'%accs[i])
    all_legends = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.legend(all_legends[:args.classnum], loc = 'upper right')
    save_file_path = os.path.join(args.ckpt, 'images', 'epoch=%d.jpg'%epoch)
    plt.savefig(save_file_path)
    plt.close()

def vis_norm_acc(norm_avg, acc_per_class, args, epoch):
    norm_sorted = np.sort(norm_avg)
    norm_idx = np.argsort(norm_avg)
    acc_reorder = acc_per_class[norm_idx]
    plt.figure(figsize=(5,5))
    plt.scatter(norm_sorted, acc_reorder)
    for idx,label in enumerate(norm_idx):
      plt.annotate(str(label), xy=(norm_sorted[idx],acc_reorder[idx]))
    plt.xlabel('Norm')
    plt.ylabel('Acc')
    save_file_path = os.path.join(args.ckpt, 'images', 'epoch=%d_norm_acc.jpg'%epoch)
    plt.savefig(save_file_path)
    plt.close()
    




def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 1, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        #if(nump_array.ndim < 3):
        #    nump_array = np.expand_dims(nump_array, axis=-1)
        #nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([33.3285]).cuda().view(1,1,1,1)
        self.std = torch.tensor([96.9255]).cuda().view(1,1,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

        
def save_checkpoint(state, args,is_best=False,filename='checkpoint.pth.tar'):
    filename = os.path.join(args.ckpt, 'checkpoints', filename)
    best_file_name = os.path.join(args.ckpt, 'checkpoints', 'model_best.pth.tar')
    torch.save(state, filename)
    if is_best:
      shutil.copyfile(filename, best_file_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def cal_class_acc(val_logits, val_target):
    class_acc = []
    pred = np.argmax(val_logits, axis=1)
    correct = (pred == val_target)
    classnum = val_logits.shape[1]
    for class_idx in range(classnum):
      curr_class = (val_target == class_idx)
      curr_corr = np.sum(curr_class*correct)
      samplenum = np.sum(curr_class)
      class_acc.append(float(curr_corr)/float(samplenum))
    return np.array(class_acc)
    
def adjust_learning_rate(optimizer,epoch, args):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    local_steps = np.copy(args.lr_steps)
    local_steps = np.insert(local_steps, 0, 0)
    local_steps = np.append(local_steps, args.epochs)
    for i in range(len(local_steps)-1): 
        if epoch >= local_steps[i] and epoch < local_steps[i+1]:
            factor = i
            break

    lr = args.base_lr*(args.lr_mults**factor)


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

    
def dist_init(port):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1,pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')

    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

    
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
        
 
def draw_plot(softmax_loss_record, regular_loss_record, iters, args):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    assert len(softmax_loss_record) == len(iters)
    axes[0].plot(iters, softmax_loss_record)
    axes[0].set_title('Softmax Loss')
    if len(regular_loss_record) > 0:
      axes[1].plot(iters, regular_loss_record)
      axes[1].set_title('Regular Loss')
    plt.savefig(os.path.join(args.ckpt, 'plots', '{}.png'.format(time.time())))

def save_record(softmax_loss_record, reg_loss_record, args):
    save_dict = {}
    save_dict['softmax_loss'] = softmax_loss_record
    save_dict['reg_loss_record'] = reg_loss_record
    savemat(os.path.join(args.ckpt, 'plots', '{}.mat').format(time.time()), save_dict)
