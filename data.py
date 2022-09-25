import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch
from PIL import Image

def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image


def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None, data_augmentation=False, root="none"):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name=='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    dataset_transform = transforms.Compose([
        *get_available_transforms(name, type, data_augmentation, root), 
        transforms.Lambda(lambda x, p=permutation: _permutate_image_pixels(x, p)),
    ])

    # load data-set
    if name == "ckplus": 
        dataset = dataset_class(
            root='<ADD_LOCAL_PATH_HERE>' + type,
            loader=lambda x: Image.open(x), 
            extensions=("png",),
            transform=dataset_transform, 
            target_transform=target_transform
        )
    elif name == "affectnet": 
        dataset = dataset_class(
            root='<ADD_LOCAL_PATH_HERE>' + type,
            loader=lambda x: Image.open(x),
            extensions=("jpg",),
            transform=dataset_transform, 
            target_transform=target_transform
        )
    elif name == "rafdb":
        dataset = dataset_class(
            root='<ADD_LOCAL_PATH_HERE>' + type,
            loader=lambda x: Image.open(x),
            extensions=("jpg",),
            transform=dataset_transform,
            target_transform=target_transform
        )
    else:
        dir = "<ADD_LOCAL_PATH_HERE>"
        dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


#----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


class TransformedDataset(Dataset):
    '''Modify existing dataset with transform; for creating multiple MNIST-permutations w/o loading data every time.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)


#----------------------------------------------------------------------------------------------------------#


# specify available data-sets.
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'ckplus': datasets.DatasetFolder,
    'affectnet': datasets.DatasetFolder,
    'rafdb': datasets.DatasetFolder
}

def get_available_transforms(dataset, mode="train", data_augmentation=False, root="none"): 
    if dataset == 'mnist': 
        return [ transforms.Pad(2), transforms.ToTensor() ]
    if dataset == 'mnist28': 
        return [ transforms.ToTensor() ]
    if dataset == 'cifar10' or dataset == 'cifar100':
        return [transforms.Grayscale(), transforms.ToTensor()]
    if dataset == 'ckplus' or dataset == 'affectnet' or dataset == 'rafdb':
        if data_augmentation == False: 
            if root == "VGG-16" or root == "MOBILENET-V2" or root == "RESNET-18" or root == "ALEXNET":
                return [
                    transforms.Resize((100, 100)), 
                    transforms.ToTensor(),
                ]            
            elif root == "none": 
                return [
                    transforms.Grayscale(), 
                    transforms.Resize((32, 32)), 
                    transforms.ToTensor(), 
                ]
        # Otherwise, if data augmentation is turned on ... 
        elif mode == 'train': 
            return [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),  
                transforms.Grayscale(), 
                transforms.Resize((32, 32)), 
                transforms.ToTensor(), 
            ]
        elif mode == 'test': 
            return [
                transforms.Grayscale(), 
                transforms.Resize((32, 32)), 
                transforms.ToTensor(), 
            ]
    raise Exception("Transforms not found!")

img_size = 100
# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 1, 'classes': 100},
    'ckplus': {'size': (img_size, img_size), 'channels': 3, 'classes': 8},
    'rafdb': {'size': (img_size, img_size), 'channels': 3, 'classes': 8},
    'affectnet': {'size': (img_size, img_size), 'channels': 3, 'classes': 8}
}


#----------------------------------------------------------------------------------------------------------#


def get_multitask_experiment(name, scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                             exception=False, split_ratio=None, data_augmentation=False, root="none"):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # prepare dataset
            train_dataset = get_dataset('mnist', type="train", permutation=None, dir=data_dir,
                                        target_transform=None, verbose=verbose)
            test_dataset = get_dataset('mnist', type="test", permutation=None, dir=data_dir,
                                       target_transform=None, verbose=verbose)
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # prepare datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class') else None
                train_datasets.append(TransformedDataset(
                    train_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
                
    elif name == 'splitMNIST':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            print("Order: ", permutation)
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))

            # prepare train and test datasets with all classes
            mnist_train = get_dataset('mnist', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            
            # NOTE: If required, take a slice from mnist_train and leave it for root pre-training. 
            if split_ratio is not None: 
                mnist_train, pretrain_dataset = torch.utils.data.random_split(mnist_train, split_ratio)

            mnist_test = get_dataset('mnist', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'splitcifar10':
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'splitcifar10' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['cifar10']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
        
            # prepare train and test datasets with all classes
            cifar10_train = get_dataset('cifar10', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
        
            # NOTE: If required, take a slice from mnist_train and leave it for root pre-training.
            if split_ratio is not None:
                cifar10_train, pretrain_dataset = torch.utils.data.random_split(cifar10_train, split_ratio)
        
            cifar10_test = get_dataset('cifar10', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario == 'domain' else None
                train_datasets.append(SubDataset(cifar10_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar10_test, labels, target_transform=target_transform))
    elif name == 'splitcifar100':
        # check for number of tasks
        if tasks > 100:
            raise ValueError("Experiment 'splitcifar100' cannot have more than 100 tasks!")
        # configurations
        config = DATASET_CONFIGS['cifar100']
        classes_per_task = int(np.floor(100 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(100))) if exception else np.random.permutation(list(range(100)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))

            # prepare train and test datasets with all classes
            cifar100_train = get_dataset('cifar100', type="train", dir=data_dir, target_transform=target_transform,
                                        verbose=verbose)

            # NOTE: If required, take a slice from mnist_train and leave it for root pre-training.
            if split_ratio is not None:
                cifar100_train, pretrain_dataset = torch.utils.data.random_split(cifar100_train, split_ratio)

            cifar100_test = get_dataset('cifar100', type="test", dir=data_dir, target_transform=target_transform,
                                       verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario == 'domain' else None
                train_datasets.append(SubDataset(cifar100_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar100_test, labels, target_transform=target_transform))
    elif name == 'splitCKPLUS': 
        # check for number of tasks
        if tasks>8:
            raise ValueError("Experiment 'splitCKPLUS' cannot have more than 8 tasks!")
        # configurations
        config = DATASET_CONFIGS['ckplus']
        classes_per_task = int(np.floor(8 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            # permutation = np.array(list(range(8))) if exception else np.random.permutation(list(range(8)))
            # print(permutation)
            # exit(0)
            permutation = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            ckplus_train = get_dataset('ckplus', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose, data_augmentation=data_augmentation, root=root)
                        
            # NOTE: If required, take a slice from mnist_train and leave it for root pre-training. 
            if split_ratio is not None: 
                ckplus_train, pretrain_dataset = torch.utils.data.random_split(ckplus_train, split_ratio)
            else:
                pretrain_dataset = None
            ckplus_test = get_dataset('ckplus', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose, data_augmentation=data_augmentation, root=root)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(ckplus_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(ckplus_test, labels, target_transform=target_transform))
    elif name == 'splitRAFDB':
        # check for number of tasks
        if tasks>7:
            raise ValueError("Experiment 'splitRAFDB' cannot have more than 7 tasks!")
        # configurations
        config = DATASET_CONFIGS['rafdb']
        classes_per_task = int(np.floor(8 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            # permutation = np.array(list(range(8))) if exception else np.random.permutation(list(range(8)))
            permutation = np.array([0, 1, 2, 3, 4, 5, 6])

            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            rafdb_train = get_dataset(name='rafdb', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose, data_augmentation=data_augmentation, root=root)
            rafdb_test = get_dataset(name='rafdb', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose, data_augmentation=data_augmentation, root=root)
            # generate labels-per-task
            labels_per_task = [
             list(np.array(range(classes_per_task)) + classes_per_task * task_id)  for task_id in range(tasks)
            ]
            print(labels_per_task)
            # if task_id < 3 else list(np.array(range(1)) + classes_per_task * task_id)
            # split them up into sub-tasks
            # pretrain_dataset = None
            # if split_ratio is not None:
            #     rafdb_train, pretrain_dataset = torch.utils.data.random_split(rafdb_train, split_ratio)
            # else:
            pretrain_dataset = None
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(rafdb_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(rafdb_test, labels, target_transform=target_transform))
    elif name == 'splitAffectNet': 
        # check for number of tasks
        if tasks>8:
            raise ValueError("Experiment 'splitAffectNet' cannot have more than 8 tasks!")
        # configurations
        config = DATASET_CONFIGS['affectnet']
        classes_per_task = int(np.floor(8 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            # permutation = np.array(list(range(8))) if exception else np.random.permutation(list(range(8)))
            permutation = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            affectnet_train = get_dataset('affectnet', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose, data_augmentation=data_augmentation, root=root)
            affectnet_test = get_dataset('affectnet', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose, data_augmentation=data_augmentation, root=root)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            pretrain_dataset = None
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(affectnet_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(affectnet_test, labels, target_transform=target_transform))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario=='domain' else classes_per_task*tasks
    # config['classes'] = classes_per_task if scenario=='domain' else 7

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task, pretrain_dataset)
