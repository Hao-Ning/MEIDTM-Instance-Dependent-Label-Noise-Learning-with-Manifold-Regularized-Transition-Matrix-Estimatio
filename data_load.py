import numpy as np
import torch.utils.data as Data
from PIL import Image
from transformer import *
import tools, pdb
from collections import Counter
from collections import defaultdict




class fmnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10, noise_type='symmetric', anchor=True):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.anchor = anchor

        original_images = np.load('./dataset/fmnist/train_images.npy')
        #original_images_error = np.load('data/mnist/train_images_test.npy')
        original_labels = np.load('./dataset/fmnist/train_labels.npy')
        original_images = np.array(original_images,dtype='uint8')
        
        self.train_data, self.train_labels,self.t = tools.dataset_split(original_images,
                                                                             original_labels, noise_rate, split_per, random_seed, num_class, noise_type,28*28)
        
        pass
    def __getitem__(self, index):

        img, label = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

     
        return img, label, index, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
   
        else:
            return len(self.val_data)
 

class fmnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
        
        self.test_data = np.load('./dataset/fmnist/test_images.npy')
        self.test_labels = np.load('./dataset/fmnist/test_labels.npy')
        self.test_data = np.array(self.test_data,dtype='uint8')
        

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)
    
class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10,noise_type='symmetric', anchor=True):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.anchor = anchor


        original_images = np.load('./dataset/cifar10/train_images.npy')
        original_labels = np.load('./dataset/cifar10/train_labels.npy')

        self.train_data, self.train_labels, self.t = tools.dataset_split(original_images,
                                                                             original_labels, noise_rate, split_per, random_seed, num_class,noise_type,32*32*3)


        print(self.train_data.shape)

        if self.anchor:
            self.train_data = self.train_data.reshape((-1,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))


    def __getitem__(self, index):
           
        if self.train:
            img, label,true = self.train_data[index], self.train_labels[index],self.t[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
            true = self.target_transform(true)

        return img, label, index, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class cifar10_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('./dataset/cifar10/test_images.npy')
        self.test_labels = np.load('./dataset/cifar10/test_labels.npy')
        self.test_data = self.test_data.reshape((10000,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
    
    def __len__(self):
        return len(self.test_data)
    
class svhn_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10,noise_type='symmetric', anchor=True):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.anchor = anchor


        original_images = np.load('./dataset/svhn/train_images.npy')
        original_labels = np.load('./dataset/svhn/train_labels.npy')

        self.train_data, self.train_labels, self.t = tools.dataset_split(original_images,
                                                                             original_labels, noise_rate, split_per, random_seed, num_class,noise_type,32*32*3)


        print(self.train_data.shape)

        if self.anchor:
            self.train_data = self.train_data.reshape((-1,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))


    def __getitem__(self, index):
           
        if self.train:
            img, label,true = self.train_data[index], self.train_labels[index],self.t[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
            true = self.target_transform(true)

        return img, label, index, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class svhn_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('./dataset/svhn/test_images.npy')
        self.test_labels = np.load('./dataset/svhn/test_labels.npy')
        self.test_data = self.test_data.reshape((-1,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
    
    def __len__(self):
        return len(self.test_data)

    
class cifar100_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=100,noise_type='symmetric', anchor=True):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.anchor = anchor

        original_images = np.load('./dataset/cifar100/train_images.npy')
        original_labels = np.load('./dataset/cifar100/train_labels.npy')

        self.train_data, self.train_labels, self.t = tools.dataset_split(original_images,
                                                                             original_labels, noise_rate, split_per, random_seed, num_class,noise_type,32*32*3)

        ind = np.arange(0,len(self.train_data))
        
        index_labels = defaultdict(int)
        labels_index = defaultdict(list)
        
        for i in range(len(self.train_data)):
            index_labels[ind[i]] = self.train_labels[i]
            labels_index[self.train_labels[i]].append(ind[i])
        
        num = 0
        ret = []
        while True:
            flag = [True for i in labels_index if len(labels_index[i][num * 8:-1]) < 8]
            if len(flag) >= 96:
                break
            indices = torch.randperm(100).tolist()
            for kid in indices:
                select_indexes = labels_index[kid]
                if len(select_indexes[num * 8:-1]) < 8:
                    ret = ret + np.random.choice(select_indexes, size=8, replace=False).tolist()
                    continue
                ret = ret + select_indexes[num * 8:(num + 1) * 8]
        
            num = num + 1
        self.train_data = self.train_data[ret]
        self.train_labels = self.train_labels[ret]
        self.t = self.t[ret]
        
        print(self.train_data.shape)
        
        if self.anchor:
            self.train_data = self.train_data.reshape((-1,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            true = self.t[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
            ture = self.target_transform(true)
     
        return img, label,index,index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)

class cifar100_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('./dataset/cifar100/test_images.npy')
        self.test_labels = np.load('./dataset/cifar100/test_labels.npy')
        self.test_data = self.test_data.reshape((10000,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)


class distilled_cifar10_dataset(Data.Dataset):
    def __init__(self, transform = None,target_transform = None):

        self.transform = transform
        self.target_transform = target_transform
        original_images = np.load('./distilled_data/cifar10/distilled_images.npy')
        original_labels = np.load('./distilled_data/cifar10/distilled_labels.npy')
        self.distilled_imgs, self.distilled_labels = original_images,original_labels


    def __getitem__(self, index):

        img,  noisy_label = self.distilled_imgs[index],   self.distilled_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            noisy_label = self.target_transform(noisy_label)

        return img,  noisy_label, index, index

    def __len__(self):
        return len(self.distilled_imgs)

class distilled_svhn_dataset(Data.Dataset):
    def __init__(self, transform = None,target_transform = None):

        self.transform = transform
        self.target_transform = target_transform
        original_images = np.load('./distilled_data/svhn/distilled_images.npy')
        original_labels = np.load('./distilled_data/svhn/distilled_labels.npy')
        self.distilled_imgs, self.distilled_labels = original_images,original_labels


    def __getitem__(self, index):

        img,  noisy_label = self.distilled_imgs[index],   self.distilled_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            noisy_label = self.target_transform(noisy_label)

        return img,  noisy_label, index, index

    def __len__(self):
        return len(self.distilled_imgs)

        
class distilled_fmnist_dataset(Data.Dataset):
    def __init__(self, transform = None,target_transform = None):

        self.transform = transform
        self.target_transform = target_transform
        original_images = np.load('./distilled_data/fmnist/distilled_images.npy')
        original_labels = np.load('./distilled_data/fmnist/distilled_labels.npy')
        self.distilled_imgs, self.distilled_labels = original_images,original_labels


    def __getitem__(self, index):

        img,  noisy_label = self.distilled_imgs[index],   self.distilled_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            noisy_label = self.target_transform(noisy_label)

        return img,  noisy_label, index, index

    def __len__(self):
        return len(self.distilled_imgs)
