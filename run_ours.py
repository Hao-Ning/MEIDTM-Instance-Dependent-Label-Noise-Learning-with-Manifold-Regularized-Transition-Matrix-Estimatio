import tools
import data_load
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import transform_train, transform_test,transform_target
from torch.optim.lr_scheduler import MultiStepLR
from models import *
import time
import heapq
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler
from random import shuffle
from torch.utils.data import Dataset,DataLoader
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--save_dir', type=str, help='dir to save model files', default='saves_Tx_manifold')
parser.add_argument('--dataset', type = str, help = 'fmnist, cifar10, and svhn', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--noise_type', type=str, default='instance')  #flip, symmetric, asymmetric,instance
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default =0.3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-3)
parser.add_argument('--anchor',default=True, action='store_false')
parser.add_argument('--warmup_epoch', type = int, default = 6) 
parser.add_argument('--lam', type = float, default =0.35)
parser.add_argument('--sigma', type = float, default =1.1)
parser.add_argument('--u', type = float, default =0.8)

args = parser.parse_args()
np.set_printoptions(precision=2,suppress=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda:'+ str(args.device))


loss_func_ce = F.nll_loss

val_loss_list = []
val_acc_list = []
test_acc_list = []


def distilling(model,train_loader,train_data):
    with torch.no_grad():
        print('Distilling start: ')
        model.eval()
        distilled_index_list = []
        distilled_dirty_index_list = []
        p_clean_list = []
        p_dirty_list = []
        
        for i,(imgs,labels,_,indexes) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            threshold = int(args.u * len(imgs))
            clean = model(imgs)

            p = torch.max(clean, dim = 1)[0]
            p_index = torch.max(clean, dim = 1)[1]

            p , index = p.sort()
            p_index = p_index[index]


            #index_clean = p_index[:threshold]

            distilled_indexes = indexes[index]
            distilled_indexes =distilled_indexes[:threshold]
            distilled_index_list.extend(distilled_indexes)


    distilled_index_list = np.array(distilled_index_list)
    distilled_imgs, distilled_labels, distilled_trues = train_data.train_data[distilled_index_list], train_data.train_labels[distilled_index_list], train_data.t[distilled_index_list]

    np.save('./distilled_data/{}/distilled_images.npy'.format(args.dataset),distilled_imgs)
    np.save('./distilled_data/{}/distilled_labels.npy'.format(args.dataset), distilled_labels)

    print('Number of distilled data: ',len(distilled_imgs))
    print('Distilling finished')
    
    if args.dataset == 'fmnist':
        train_distill_data = data_load.distilled_fmnist_dataset(transform=transform_train(args.dataset),
                                                               target_transform=transform_target)
    if args.dataset == 'cifar10':
        train_distill_data = data_load.distilled_cifar10_dataset(transform=transform_train(args.dataset),
                                                                 target_transform=transform_target)
    if args.dataset == 'svhn':
        train_distill_data = data_load.distilled_svhn_dataset(transform=transform_train(args.dataset),
                                                                 target_transform=transform_target)
    
    train_distill_loader = DataLoader(dataset=train_distill_data,
                                      batch_size=args.batch_size,
                                      num_workers=4,
                                      shuffle=True,
                                      drop_last=True)
    return train_distill_loader
    


def warmup(warmup_loader,model,optimizer):
    train_loss = 0.
    train_acc = 0.
    data_num = 0.
    for batch_i, (batch_x, batch_y,indexes,_) in enumerate(warmup_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
                
        ind = torch.randperm(len(batch_x)).tolist()
        batch_x = batch_x[ind]
        batch_y = batch_y[ind]
        
        optimizer.zero_grad()

        clean = model(batch_x)
        ce_loss = loss_func_ce(clean.log(), batch_y.long())
        res = torch.mean(torch.sum(clean.log() * clean, dim=1))
        data_num = data_num + len(clean)

        loss = ce_loss +  res

        train_loss += loss.item() * clean.shape[0]
        

        pred = torch.max(clean, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()

        loss.backward()
        
        optimizer.step()
    print('Warmup Loss: {:.6f},   Acc: {:.6f}'.format(
        train_loss / data_num, train_acc / data_num))


def loss_our(epoch,clean,out,batch_y,T):
    s1 = clean.clone().reshape(1, clean.shape[0], args.num_classes)
    s2 = clean.clone().reshape(clean.shape[0], 1, args.num_classes)
    ind = torch.where((torch.max(s1, dim = 2).indices - torch.max(s2, dim = 2).indices) == 0)
    s_ij =  -torch.ones(clean.shape[0], clean.shape[0]).to(device)
    #s_ij = torch.zeros(clean.shape[0], clean.shape[0]).to(device)
    s_ij[ind] = 1
    s_ij1 = torch.exp((-torch.sum((s1 - s2) ** 2, dim = 2)) / (2 * (args.sigma ** 2)))

    s_ij = s_ij * s_ij1

    T1 = T.clone().view(clean.shape[0], -1).reshape(1, clean.shape[0],
                                                    args.num_classes * args.num_classes)
    T2 = T.clone().view(clean.shape[0], -1).reshape(clean.shape[0], 1,
                                                    args.num_classes * args.num_classes)
    ij_dist = torch.sum((T1 - T2) ** 2, dim = 2)

    manifold_loss = torch.mean(s_ij.detach() * ij_dist)
    
    ce_loss = loss_func_ce(out.log(), batch_y)
    
    return ce_loss, manifold_loss

def train_our(epoch,train_loader,model,optimizer,scheduler,trans,optimizer_trans,scheduler_trans):
    train_loss = 0.
    train_acc = 0.
    data_num = 0.
    model.train()
    trans.train()
    for batch_i, (batch_x, batch_y,indexes,_) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        data_num = data_num + len(batch_x)

        clean = model(batch_x)
        T = trans(clean) 
        out = torch.bmm(clean.unsqueeze(1), T).squeeze(1)
        ce_loss, manifold_loss = loss_our(epoch,clean,out,batch_y.long(),T)
        
        loss = ce_loss + args.lam * manifold_loss
        
        train_loss += loss.item() * clean.shape[0]

        pred = torch.max(clean, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        
        optimizer_trans.zero_grad()
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        optimizer_trans.step()

    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / data_num, train_acc / data_num))
    scheduler.step()
    
    
def test(test_loader,model):
    eval_loss = 0.
    eval_acc = 0.
    data_num = 0
    with torch.no_grad():
        model.eval()

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            clean = model(batch_x)

            data_num = data_num + len(clean)
            loss = loss_func_ce(clean.log(), batch_y.long())
            eval_loss += loss.item()*len(clean)
            pred = torch.max(clean, 1)[1]
            eval_correct = (pred == batch_y).sum()
            eval_acc += eval_correct.item()

        print('Test Loss: {:.6f}, Acc: {:.9f}'.format(eval_loss / data_num,
                                                      eval_acc / data_num))

    return eval_acc / data_num


def main():
    print('noise_rate : ',args.noise_rate)
    print(args)
    
    if args.dataset == 'fmnist':
    
        args.n_epoch = 20
        args.num_classes = 10
        milestones = [20,40]
        args.warmup_epoch = 0
    
        train_data = data_load.fmnist_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                             noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type,anchor=args.anchor)
        test_data = data_load.fmnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        model = ResNet18(args.num_classes)
        trans = sig_t(device, args.num_classes)
        optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr/2, weight_decay=0)
        
        train_loader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)
    
        test_loader = DataLoader(dataset=test_data,
                           batch_size=args.batch_size,
                           num_workers=4,
                           drop_last=False)

    
    if args.dataset == 'cifar10':
        args.n_epoch = 50
        args.warmup_epoch = 10
        args.num_classes = 10
        milestones = [30,45] 
    
        train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                             noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
        test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        model = ResNet34(args.num_classes)
        trans = sig_t(device, args.num_classes,args.batch_size)
        optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr/2, weight_decay=0)
        
        train_loader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)
    
        test_loader = DataLoader(dataset=test_data,
                           batch_size=args.batch_size,
                           num_workers=4,
                           drop_last=False)
    

    if args.dataset == 'svhn':
        args.n_epoch = 50
        args.warmup_epoch = 10
        args.num_classes = 10
        milestones = [30,45]
    
        train_data = data_load.svhn_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                             noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
        test_data = data_load.svhn_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        model = ResNet34(args.num_classes)
        trans = sig_t(device, args.num_classes,args.batch_size)
        optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr/2, weight_decay=0)
        
        train_loader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)
    
        test_loader = DataLoader(dataset=test_data,
                           batch_size=args.batch_size,
                           num_workers=4,
                           drop_last=False)         
    
    model = torch.nn.DataParallel(model, device_ids=[1,2])
    model = model.to(device)
    trans = trans.to(device)
    
    #optimizer and StepLR
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scheduler_trans = MultiStepLR(optimizer_trans, milestones=milestones, gamma=0.1)
    
    best_acc = 0

    for epoch in range(args.warmup_epoch):
        print('epoch[{}], Warmup:'.format(epoch+ 1))
        print('manifold dataset : ', args.dataset)
        warmup(train_loader,model,optimizer)

    for epoch in range(args.n_epoch):
        model.train()
        #distill data 
        if epoch % 10 == 0:
            train_distilled_loader = distilling(model,train_loader,train_data) 
        print('epoch[{}], Train:'.format(epoch+ 1))
        print('manifold dataset : ', args.dataset)
        train_our(epoch,train_distilled_loader,model,optimizer,scheduler,trans,optimizer_trans,scheduler_trans) 
        
        eval_acc = test(test_loader,model)

        if eval_acc >= best_acc:
            best_acc = eval_acc

    print("Best test accuracy acc: %f" % best_acc)


if __name__=='__main__':
    main()

