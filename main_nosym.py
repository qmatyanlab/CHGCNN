import time
import torch, shutil, argparse
import torch.optim as optim
import numpy as np
import wandb
from torch_geometric.nn import to_hetero
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from data.generate_nosym import InMemoryCrystalHypergraphDataset
from model.chgcnn import CrystalHypergraphConv
from data.hypergraph.hypergraph import *
import torch_geometric.transforms as T



from random import sample


def ClassificationAccuracy(output, target):
    prediction = torch.argmax(output, dim=1)
    acc = (prediction == target).float().mean()
    return acc


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(model, device, train_loader, loss_criterion, accuracy_criterion, optimizer, epoch, task, target_name, normalizer, scheduler):
    batch_time = AverageMeter('Batch', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    accus = AverageMeter('Accu', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accus],
        prefix='Epoch: [{}]'.format(epoch))

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):

        data_time.update(time.time() - end)
        data = data.to(device, non_blocking=True)
        output = torch.squeeze(model(data))
        if task == 'regression':
            target = torch.tensor([float(i) for i in data[target_name]]).to(device)
            target_norm = normalizer.norm(target)

            target = torch.squeeze(target)
            target_norm = torch.squeeze(target_norm)

            loss = loss_criterion(output, target_norm)
            accu = accuracy_criterion(normalizer.denorm(output), target)
            losses.update(loss.item(), target_norm.size(0))
            accus.update(accu.item(), target.size(0))
        else:
            target = data[target_name].long()
            loss = loss_criterion(output, target)
            accu = accuracy_criterion(output, target.float())
            losses.update(loss.item(), target.size(0))
            accus.update(accu.item(), target.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
    if task == 'regression':
        wandb.log({'train-mse-avg': losses.avg, 'train-mae-avg': accus.avg, 'epoch': epoch, 'batch-time': batch_time.avg, 'lr': scheduler.get_last_lr()}) 
    else:
        wandb.log({'train-nll-avg': losses.avg, 'train-accuracy-avg': accus.avg, 'epoch': epoch, 'batch-time': batch_time.avg}) 
    return losses.avg, accus.avg


def validate(model, device, test_loader, loss_criterion, accuracy_criterion, epoch, task, target_name, normalizer):
    batch_time = AverageMeter('Batch', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    accus = AverageMeter('Accu', ':.4f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, accus],
        prefix='Val: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(test_loader):

            data = data.to(device, non_blocking=True)
            output = torch.squeeze(model(data))
            if task == 'regression':
                target = torch.tensor([float(i) for i in data[target_name]]).to(device)
                target_norm = normalizer.norm(target)
                target = torch.squeeze(target)
                target_norm = torch.squeeze(target_norm)

                loss = loss_criterion(output, target_norm)
                accu = accuracy_criterion(normalizer.denorm(output), target)
                losses.update(loss.item(), target_norm.size(0))
                accus.update(accu.item(), target.size(0))
            else:
                target = data[target_name].long()
                loss = loss_criterion(output, target)
                accu = accuracy_criterion(output, target)
                losses.update(loss.item(), target.size(0))
                accus.update(accu.item(), target.size(0))


            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
    if task == 'regression':
        wandb.log({'val-mse-avg': losses.avg, 'val-mae-avg': accus.avg, 'i': i, 'batch-time': batch_time.avg, 'epoch': epoch}) 
    else:
        wandb.log({'val-nll-avg': losses.avg, 'val-accu-avg': accus.avg, 'i': i, 'batch-time': batch_time.avg, 'epoch': epoch}) 
    return accus.avg


def main():
    parser = argparse.ArgumentParser(description='CGCNN')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='trainning ratio')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 5e-5)',
                        dest='weight_decay')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    parser.add_argument('--optim', default='SGD', type=str, metavar='Adam',
                        help='choose an optimizer, SGD or Adam, (default:Adam)')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--task', default='regression', type=str)
    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--drop-last', default=False, type=bool)
    parser.add_argument('--pin-memory', default=False, type=bool)
    parser.add_argument('--dir', default='dataset', type=str)
    parser.add_argument('--normalize', default=True, type=bool)
    parser.add_argument('--target', default = 'form_en', type=str, help='formation energy (form_en), band gap (band_gap) or energy above hull (en_abv_hull) prediction task') 
    parser.add_argument('--scheduler', default=True, type=bool,
            help = 'use scheduler')

    args = parser.parse_args()

    best_accu = 1e6
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)


    torch.backends.cudnn.benchmark = True
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="chgcnn",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "TransformerConv" ,
        "dataset": "Formation Energy per Atom",
        "epochs": args.epochs,
        "batch_size": args.batch_size
        }
    )

    #### Create dataset
    print(f'Finding data in {args.dir}...')
    dataset = InMemoryCrystalHypergraphDataset(args.dir)

        
    #### Initiliaze model 
    print('Initializing model...') 
    if args.task == 'classification':
        class_bool = True
    else:
        class_bool = False
    model = CrystalHypergraphConv(classification = class_bool).to(device)

    
    #### Divide data into train and test sets
    n_data = len(dataset)
    train_split = int(n_data * args.train_ratio)
    dataset_train, dataset_val = random_split(
        dataset,
        [train_split, len(dataset) - train_split],
        generator=torch.Generator().manual_seed(args.seed)
    )

    #### Load data into DataLoaders/Batches
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=False, #args.pin_memory,
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory
    )

    #### Set target as y
    if args.target == 'form_en':
        print(f'Setting target as formation energy (eV/atom)')
    elif args.target == 'band_gap':
        print(f'Setting target as band gap (eV)')
    elif args.target == 'en_abv_hull':
        print(f'Setting target as energy above hull (eV)')
    elif args.target == 'metalicity':
        args.task = 'classification'
        args.num_class = 2
        args.normalize = False
        print(f'Setting target as metalicity (metal 1/nm 0)')
    else:
        print(f'Target {args.target} not found!')

    #### Set normalizer (for targets)
    print(args.normalize)
    if args.normalize == True:
        if len(dataset) < 1000:
            print(dataset[0][args.target])
            sample_targets = [torch.tensor(float(dataset[i][args.target])) for i in range(len(dataset))]
        else:
            sample_targets = [torch.tensor(float(dataset[i][args.target])) for i in sample(range(len(dataset)), 1000)]
        normalizer = Normalizer(sample_targets)
        print('normalizer initialized!')
    else:
        normalizer = None


    #### Set optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print(f'SGD chosen as optimizer with lr {args.lr}, mom {args.momentum}, wd {args.weight_decay}')
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f'Adam chosen as optimizer with lr {args.lr}, wd {args.weight_decay}')

    else:
        raise ValueError('Optimizer must be SGD or Adam.')

    #### Set up scheduler
    if args.scheduler == True:
        scheduler = CosineAnnealingLR(optimizer, T_max = 300, verbose = True)

    #### Set cost and loss functions 
    if args.task == 'regression':
        loss_criterion = torch.nn.MSELoss()
        accuracy_criterion = torch.nn.L1Loss()
        print('Using MSE accuracy and L1 for training loss')
    elif args.num_class == 2 and args.task != 'regression':
        loss_criterion = torch.nn.CrossEntropyLoss()
        accuracy_criterion = ClassificationAccuracy
        print('Using cross entropy for training loss and basic accuracy')
    else:
        loss_criterion = torch.nn.CrossEntropyLoss()
        accuracy_criterion = torch.nn.CrossEntropyLoss()
        print('Using cross entropy for training loss and accuracy')

    #### Resume mechanism
    if args.resume:
        print("=> loading checkpoint")
        checkpoint = torch.load('checkpoint.pth.tar')
        args.start_epoch = checkpoint['epoch'] + 1
        best_accu = checkpoint['best_accu']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    #### Loop through train and test for set number of epochs
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_accu = train(model, device, train_loader, loss_criterion, accuracy_criterion, optimizer,
                                       epoch, args.task, args.target, normalizer, scheduler)
        val_accu = validate(model, device, val_loader, loss_criterion, accuracy_criterion, epoch, args.task, args.target, normalizer)

        is_best = train_accu < best_accu
        best_accu = min(train_accu, best_accu)

        scheduler.step()
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_accu': best_accu,
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }, is_best)



    ckpt = torch.load('model_best.pth.tar')
    print(ckpt['best_accu'])
    wandb.finish()

#### Define normalizer for target data (from CGCNN source code)
class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor_list):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(torch.stack(tensor_list,dim=0))
        self.std = torch.std(torch.stack(tensor_list,dim=0))

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
