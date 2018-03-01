import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb

import time

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0

'''
def im_show(name, image, resize=1):
    H,W = image.shape[0:2]
    #cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    #cv2.imshow('test',image)
    #plt.imshow(image)
    #plt.show()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(pop[0]) + 'im.jpg', image)
    #plt.savefig(str(pop) + 'im.jpg')
    pop[0] = pop[0] + 1
    #cv2.resizeWindow(name, round(resize*W), round(resize*H))


def tensor_to_img(img, mean=0, std=1):
    #print (img.numpy().shape)
    img = np.transpose(img.numpy(), (1, 2, 0))
    #cv2.imwrite(str(pop[0])+ str(pop[0]) + 'im.jpg', img)
    #plt.savefig(str(pop) + str(pop) + 'im.jpg')
    img = (img*std + mean)*255
    img = img.astype(np.uint8)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img
'''

def plot_stats(epoch, data_1, data_2, label_1, label_2, plt):
    plt.plot(range(epoch), data_1, 'r--', label=label_1)
    plt.plot(range(epoch), data_2, 'g--', label=label_2)
    plt.legend()
    
def logPolar_transform(image):
    M = 44.2244 # for 224 * 224 sized image
    img = np.array(image)
    return cv2.logPolar(img, (img.shape[0]/2, img.shape[1]/2), M, cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

def main():
    run_time = time.ctime().replace(' ', '_')[:-8] 
    directory = 'progress/' + run_time
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory + '/logs.txt', 'w')
    global args, best_prec1
    print ("GPU processing available : ", torch.cuda.is_available())
    print ("Number of GPU units available :", torch.cuda.device_count())
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    train_prec1_plot = []
    train_prec5_plot = []
    train_loss_plot = []
    val_prec1_plot = []
    val_prec5_plot = []
    val_loss_plot = []
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_prec1_plot = train_prec1_plot + checkpoint['train_prec1_plot']
            train_prec5_plot = train_prec5_plot + checkpoint['train_prec5_plot']
            train_loss_plot = train_loss_plot + checkpoint['train_loss_plot']
            val_prec1_plot = val_prec1_plot + checkpoint['val_prec1_plot']
            val_prec5_plot = val_prec5_plot + checkpoint['val_prec5_plot']
            val_loss_plot = val_loss_plot + checkpoint['val_loss_plot']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            #print("prahal...", train_prec1_plot, train_prec5_plot, train_loss_plot)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.Lambda(lambda x: logPolar_transform(x)),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            #transforms.Lambda(lambda x: logPolar_transform(x)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, f)
        return

    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_prec1, train_prec5, train_loss  = train(train_loader, model, criterion, optimizer, epoch, f)
        train_prec1_plot.append(train_prec1)
        train_prec5_plot.append(train_prec5)
        train_loss_plot.append(train_loss)
        
        # evaluate on validation set
        val_prec1, val_prec5, val_loss = validate(val_loader, model, criterion, f)
        val_prec1_plot.append(val_prec1)
        val_prec5_plot.append(val_prec5)
        val_loss_plot.append(val_loss)
        
        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        save_checkpoint({
            'train_prec1_plot':train_prec1_plot,
            'train_prec5_plot':train_prec5_plot,
            'train_loss_plot':train_loss_plot,
            'val_prec1_plot':val_prec1_plot,
            'val_prec5_plot':val_prec5_plot,
            'val_loss_plot':val_loss_plot,
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        
        #plot data
        plt.figure(figsize=(12,12))
        plt.subplot(3,1,1)
        plot_stats(epoch+1, train_loss_plot, val_loss_plot, 'train_loss', 'val_loss', plt)
        plt.subplot(3,1,2)
        plot_stats(epoch+1, train_prec1_plot, val_prec1_plot, 'train_prec1', 'val_prec1', plt)
        plt.subplot(3,1,3)
        plot_stats(epoch+1, train_prec5_plot, val_prec5_plot, 'train_prec5', 'val_prec5', plt)
        plt.savefig('progress/' + run_time + '/stats.jpg')
        plt.clf()
    f.close()

def train(train_loader, model, criterion, optimizer, epoch, f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    #classes = dataset.classes
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        '''
        This is for testing purposes and render log polar
        for n in range(len(input)):
            image=input[n]
            label=target[n]
            print('\t\tlabel=%d'%(label))
            #print('')

            im_show('image', tensor_to_img(image), resize=6 )
            #cv2.waitKey(1)
        break
        '''

        target = target.cuda(async=True)
        #input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #a = list(model.parameters())[0] 
        loss.backward()
        optimizer.step()
        #b = list(model.parameters())[0] 
        #print ("Prahal", torch.equal(a.data, b.data), len(list(model.parameters())))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress_stats = 'Time: {0} Epoch: [{1}][{2}/{3}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   time.ctime()[:-8], epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5)
            print(progress_stats)
            f.write(progress_stats + "\n")
            f.flush()
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        #input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        #pdb.set_trace()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
               i, len(val_loader), batch_time=batch_time, loss=losses,
               top1=top1, top5=top5))

    val_stats = 'VALIDATION Time {time} * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.4f}'.format(
        time=time.ctime()[:-8],top1=top1, top5=top5, loss=losses)
    print(val_stats)
    f.write(val_stats + "\n")
    f.flush()
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    #print ("Prahallllll, output", output.size())
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print('Correct', correct.size(), target.size(), target.view(1, -1).size(), pred.size())
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    #time.sleep(10)
    return res


if __name__ == '__main__':
    main()
