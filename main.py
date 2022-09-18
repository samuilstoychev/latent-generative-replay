#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import torch
from torch import optim
import utils
import evaluate
from data import get_multitask_experiment
from encoder import Classifier
from vae_models import AutoEncoder
from train import train_cl
from continual_learner import ContinualLearner
from replayer import Replayer
from param_values import set_default_values
# New imports 
from train import pretrain_root
from train import pretrain_baseline
from autoencoder_latent import AutoEncoderLatent
from train_latent import train_cl_latent
import evaluate_latent
from root_classifier import RootClassifier
from top_classifier import TopClassifier
from cnn_classifier import CNNClassifier
from cnn_root_classifier import CNNRootClassifier
from cnn_top_classifier import CNNTopClassifier
from cl_metrics import RAMU
from cl_metrics import CPUUsage
from cl_metrics import GPUUsage
from cl_metrics import MAC
import gc
import pickle
from pretrained_root_classifier import PretrainedRootClassifier

parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST', 'splitCKPLUS', 'splitRAFDB', 'splitAffectNet',
                                                                                  'splitcifar10', 'splitcifar100'])
task_params.add_argument('--scenario', type=str, default='class', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")
loss_params.add_argument('--bce-distill', action='store_true', help='distilled loss on previous classes for new'
                                                                    ' examples (only if --bce & --scenario="class")')

# model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                   " (instead of a 'multi-headed' one)")

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--z-dim', type=int, default=100, help='size of latent representation (default: 100)')
replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'naive-rehearsal']
replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
replay_params.add_argument('--distill', action='store_true', help="use distillation for replay?")
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
replay_params.add_argument('--agem', action='store_true', help="use gradient of replay as inequality constraint")
# -generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--g-z-dim', type=int, default=100, help='size of latent representation (default: 100)')
genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
# - hyper-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--g-iters', type=int, help="# batches to train generator (default: as classifier)")
gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: lr)")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
cl_params.add_argument('--lambda', type=float, dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
cl_params.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
cl_params.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
cl_params.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--xdg', action='store_true', help="Use 'Context-dependent Gating' (Masse et al, 2018)")
cl_params.add_argument('--gating-prop', type=float, metavar="PROP", help="--> XdG: prop neurons per layer to gate")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")

# latent replay 
latent_params = parser.add_argument_group('Latent Replay')
latent_params.add_argument('--latent-replay', type=str, default='off', choices=['on', 'off'])
latent_params.add_argument('--network', type=str, default="mlp", choices=['mlp', 'cnn'])
latent_params.add_argument('--latent-size', type=int, default=200)
latent_params.add_argument('--pretrain-baseline', action='store_true')
latent_params.add_argument('--out-channels', type=int, default=5)
latent_params.add_argument('--kernel-size', type=int, default=5)
latent_params.add_argument('--pretrain-iters', type=int, default=1000)
latent_params.add_argument('--data-augmentation', action='store_true')
latent_params.add_argument('--pretrained-root', type=str, default="none", choices=['none', 'VGG-16', "MOBILENET-V2", "RESNET-18", "ALEXNET"])
latent_params.add_argument('--buffer-size', type=int, default=1000)
latent_params.add_argument('--early-stop', action='store_true')
latent_params.add_argument('--validation', action='store_true')
latent_params.add_argument('--identifier', type=str, default="na")

ramu = RAMU()
cpuu = CPUUsage()

def run(args, verbose=False):

    print("RAM AT BEGINNING:", ramu.compute("BEGINNING"))
    print("Latent replay turned", args.latent_replay)
    # Set default arguments & check for incompatible options
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -if XdG is selected but not the Task-IL scenario, give error
    if (not args.scenario=="task") and args.xdg:
        raise ValueError("'XdG' is only compatible with the Task-IL scenario.")

    # -if XdG is selected together with both replay and EWC, give error (either one of them alone with XdG is fine)
    if (args.xdg and args.gating_prop>0) and (not args.replay=="none") and (args.ewc or args.si):
        raise NotImplementedError("XdG is not supported with both '{}' replay and EWC / SI.".format(args.replay))
        #--> problem is that applying different task-masks interferes with gradient calculation
        #    (should be possible to overcome by calculating backward step on EWC/SI-loss also for each mask separately)
    # -if 'BCEdistill' is selected for other than scenario=="class", give error
    if args.bce_distill and not args.scenario=="class":
        raise ValueError("BCE-distill can only be used for class-incremental learning.")
    # -create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)

    scenario = args.scenario
    # If Task-IL scenario is chosen with single-headed output layer, set args.scenario to "domain"
    # (but note that when XdG is used, task-identity information is being used so the actual scenario is still Task-IL)
    if args.singlehead and args.scenario=="task":
        scenario="domain"

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#
    print("RAM BEFORE LOADING DATA:", ramu.compute("BEFORE LOADING"))
    # Prepare data for chosen experiment
    if verbose:
        print("\nPreparing the data...")

    if args.experiment == "splitAffectNet": 
        split_ratio = None 
    elif args.experiment=="splitCKPLUS": 
        split_ratio = None
    elif args.experiment=="splitRAFDB":
        split_ratio = None
    elif args.experiment == "splitcifar10":
        split_ratio = [40000, 10000]
    elif args.experiment == "splitcifar100":
        split_ratio = [45000, 5000]
    else:
        split_ratio = [50000, 10000]
    print("SPLIT RATIO:", split_ratio)

    if args.experiment == "splitCKPLUS": 
        dataset = "ckplus"
    elif args.experiment == "splitRAFDB":
        dataset = "rafdb"
    elif args.experiment == "splitAffectNet": 
        dataset = "affectnet"
    elif args.experiment == "splitcifar10":
        dataset = "cifar10"
    elif args.experiment == "splitcifar100":
        dataset = "cifar100"
    else: 
        dataset = "mnist"

    (train_datasets, test_datasets), config, classes_per_task, pretrain_dataset = get_multitask_experiment(
        name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir,
        verbose=verbose, exception=True if args.seed==0 else False, split_ratio=split_ratio, 
        data_augmentation=args.data_augmentation, root=args.pretrained_root
    )
    
    print("RAM AFTER LOADING DATA:", ramu.compute("RAM AFTER LOADING DATA"))

    #----------------------------------------------------------#
    #-----DEFINING THE ROOT AND TOP (FOR LATENT REPLAY) -------#
    #----------------------------------------------------------#

    if args.latent_replay == "on": 
        if args.network == "mlp": 
            root_model = RootClassifier(
                image_size=config['size'], image_channels=config['channels'], classes=config['classes'], 
                fc_layers=args.fc_lay, fc_units=args.fc_units,
                fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, 
                dataset=dataset
            ).to(device)
        elif args.network == "cnn": 
            root_model = CNNRootClassifier(image_size=config['size'], classes=config['classes'], 
                                           latent_space=args.latent_size, out_channels=args.out_channels, 
                                           kernel_size=args.kernel_size, 
                                           dataset=dataset).to(device)

        root_model.optim_list = [{'params': filter(lambda p: p.requires_grad, root_model.parameters()), 'lr': args.lr}]
        root_model.optim_type = args.optimizer
        if root_model.optim_type in ("afdam", "adam_reset"):
            root_model.optimizer = optim.Adam(root_model.optim_list, betas=(0.9, 0.999))
        elif root_model.optim_type=="sgd":
            root_model.optimizer = optim.SGD(root_model.optim_list)
        if args.network == "mlp": 
            top_model = TopClassifier(classes=config['classes'], 
                fc_layers=args.fc_lay, fc_units=args.fc_units, 
                fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
            ).to(device)
        elif args.network == "cnn": 
            top_model = CNNTopClassifier(classes=config['classes'], latent_space=args.latent_size).to(device)
        top_model.optim_list = [{'params': filter(lambda p: p.requires_grad, top_model.parameters()), 'lr': args.lr}]
        top_model.optim_type = args.optimizer
        if top_model.optim_type in ("adam", "adam_reset"):
            top_model.optimizer = optim.Adam(top_model.optim_list, betas=(0.9, 0.999))
        elif top_model.optim_type=="sgd":
            top_model.optimizer = optim.SGD(top_model.optim_list)

    #-------------------------------------------------------------------------------------------------#

    #------------------------------#
    #----- MODEL (CLASSIFIER) -----#
    #------------------------------#
    print("RAM BEFORE CLASSIFER:", ramu.compute("BEFORE CLASSIFIER"))

    if args.pretrained_root != "none": 
        model = PretrainedRootClassifier(
            classes=config['classes'], latent_space=args.latent_size, 
            binaryCE=args.bce, binaryCE_distill=args.bce_distill, AGEM=args.agem,
            out_channels=args.out_channels, kernel_size=args.kernel_size, root_type=args.pretrained_root
        ).to(device)
        print("RAM AFTER CLASSIFER:", ramu.compute("AFTER CLASSIFIER"))
    elif args.network == "cnn": 
        model = CNNClassifier(
            image_size=config['size'], classes=config['classes'], latent_space=args.latent_size, 
            binaryCE=args.bce, binaryCE_distill=args.bce_distill, AGEM=args.agem,
            out_channels=args.out_channels, kernel_size=args.kernel_size, 
            dataset=dataset
        ).to(device)
        print("RAM AFTER CLASSIFER:", ramu.compute("AFTER CLASSIFIER"))
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
            fc_bn=True if args.fc_bn=="yes" else False, excit_buffer=True if args.xdg and args.gating_prop>0 else False,
            binaryCE=args.bce, binaryCE_distill=args.bce_distill, AGEM=args.agem, 
            dataset=dataset
        ).to(device)
        print("RAM AFTER CLASSIFER:", ramu.compute("AFTER CLASSIFIER"))

    # Define optimizer (only include parameters that "requires_grad")
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type in ("adam", "adam_reset"):
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type=="sgd":
        model.optimizer = optim.SGD(model.optim_list)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))

    #-------------------------------------------------------------------------------------------------#

    #------------------------------#
    #----- ROOT PRETRAINING -------#
    #------------------------------#
    print("RAM BEFORE PRE-TRAINING", ramu.compute("BEFORE PRETRAINING"))
    if args.latent_replay == "on": 
        if args.pretrained_root != "none": 
            root_model = model.feature_extractor
        else: 
            pretrain_root(root_model, model, pretrain_dataset, n_classes=config['classes'], 
            batch_iterations=args.pretrain_iters)
            del pretrain_dataset
            gc.collect()
    elif args.pretrain_baseline and args.pretrained_root == "none":
        pretrain_baseline(model, pretrain_dataset, n_classes=config['classes'], 
        batch_iterations=args.pretrain_iters)
        del pretrain_dataset
        gc.collect()

    print("RAM AFTER PRE-TRAINING", ramu.compute("AFTER PRETRAINING"))

    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------#
    #----- CL-STRATEGY: ALLOCATION -----#
    #-----------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        if args.ewc:
            model.fisher_n = args.fisher_n
            model.gamma = args.gamma
            model.online = args.online
            model.emp_FI = args.emp_fi

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner):
        model.si_c = args.si_c if args.si else 0
        if args.si:
            model.epsilon = args.epsilon

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and (args.xdg and args.gating_prop>0):
        mask_dict = {}
        excit_buffer_list = []
        for task_id in range(args.tasks):
            mask_dict[task_id+1] = {}
            for i in range(model.fcE.layers):
                layer = getattr(model.fcE, "fcLayer{}".format(i+1)).linear
                if task_id==0:
                    excit_buffer_list.append(layer.excit_buffer)
                n_units = len(layer.excit_buffer)
                gated_units = np.random.choice(n_units, size=int(args.gating_prop*n_units), replace=False)
                mask_dict[task_id+1][i] = gated_units
        model.mask_dict = mask_dict
        model.excit_buffer_list = excit_buffer_list


    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, Replayer):
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp
        if args.latent_replay == "on": 
            top_model.replay_targets = "soft" if args.distill else "hard"
            top_model.KD_temp = args.temp

    print("RAM BEFORE GENERATOR:", ramu.compute("BEFORE GENERATOR"))
    # If needed, specify separate model for the generator
    train_gen = True if args.replay=="generative" else False
    if train_gen:
        # -specify architecture
        if args.latent_replay == "on": 
            generator = AutoEncoderLatent(
                latent_size=args.latent_size,
                fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.g_z_dim, classes=config['classes'],
                fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
            ).to(device)
        else: 
            gen_input_size = (100, 100) if args.pretrained_root != "none" else config['size'] 
            gen_channels = 3 if args.pretrained_root != "none" else config['channels'] 

            generator = AutoEncoder(
                image_size=gen_input_size, image_channels=gen_channels,
                fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.g_z_dim, classes=config['classes'],
                fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
                dataset=dataset
            ).to(device)
        # -set optimizer(s)
        generator.optim_list = [{'params': filter(lambda p: p.requires_grad, generator.parameters()), 'lr': args.lr_gen}]
        generator.optim_type = args.optimizer
        if generator.optim_type in ("adam", "adam_reset"):
            generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(generator.optim_list)
    else:
        generator = None
    
    print("RAM AFTER DECLARING GENERATOR:", ramu.compute("AFTER GENERATOR"))

    mac = MAC()
    if args.experiment == "splitCKPLUS" or args.experiment == "splitAffectNet" or args.experiment == 'splitRAFDB':
        if args.pretrained_root != "none":
            dummy_data = torch.rand(32, 3, 100, 100)
        else: 
            dummy_data = torch.rand(32, 1, config["size"][0], config["size"][1])
    else: 
        dummy_data = torch.rand(32, 1, config["size"], config["size"])
    dummy_data = dummy_data.to(device)

    if args.latent_replay == "on": 
        if args.pretrained_root == "none": 
            print("MACs of root classifier", mac.compute(root_model, dummy_data))
        print("MACs of top classifier:", mac.compute(top_model, root_model(dummy_data)))
    else: 
        print("MACs of model:", mac.compute(model, dummy_data))

    print("RAM BEFORE REPORTING:", ramu.compute("BEFORE REPORTING"))

    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        if args.latent_replay == "on": 
            utils.print_model_info(top_model, title="TOP")
            if args.pretrained_root != "none":
                print("ROOT = " + args.pretrained_root)
                # utils.print_model_info(root_model, title="ROOT= "+str(args.pretrained_root))
            else: 
                utils.print_model_info(root_model, title="ROOT")
        else: 
            utils.print_model_info(model, title="MAIN MODEL")
        # -generator
        if generator is not None:
            utils.print_model_info(generator, title="GENERATOR")

    if args.latent_replay == "on":
        del model
        gc.collect()

    print("RAM BEFORE TRAINING:", ramu.compute("BEFORE TRAINING"))
    print("CPU BEFORE TRAINING:", cpuu.compute("BEFORE TRAINING"))
    if cuda: 
        print("INITIALISING GPU TRACKER")
        gpuu = GPUUsage(0)

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if verbose:
        print("\nTraining...")
    # Keep track of training-time
    start = time.time()
    # Train model
    if args.latent_replay == "on": 
        res = train_cl_latent(
            top_model, train_datasets, root=root_model, replay_mode=args.replay, scenario=scenario, classes_per_task=classes_per_task,
            iters=args.iters, batch_size=args.batch,
            generator=generator, gen_iters=args.g_iters,
            buffer_size=args.buffer_size, valid_datasets=test_datasets if (args.early_stop or args.validation) else None, 
            early_stop=args.early_stop, validation=args.validation,
            plot=True
        )
    else: 
        res = train_cl(
            model, train_datasets, replay_mode=args.replay, scenario=scenario, classes_per_task=classes_per_task,
            iters=args.iters, batch_size=args.batch,
            generator=generator, gen_iters=args.g_iters,
            buffer_size=args.buffer_size, valid_datasets=test_datasets if (args.early_stop or args.validation) else None, 
            early_stop=args.early_stop, validation=args.validation
        )
    if res is not None and args.validation: 
        # Store the validation data if required
        with open('validation_{}/val_{}_{}_{}.pkl'.format(args.scenario, dataset, args.seed, args.identifier), 'wb') as output: 
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
            
    if cuda:
        print("GPU BEFORE EVALUATION:", gpuu.compute("BEFORE EVALUATION"))

    training_time = time.time() - start

    print("RAM BEFORE EVALUATION:", ramu.compute("BEFORE EVALUATION"))
    print("CPU BEFORE EVALUATION:", cpuu.compute("BEFORE EVALUATION"))

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
    if args.latent_replay == "on": 
        precs = [evaluate_latent.validate(
            top_model, test_datasets[i], root=root_model, verbose=False, test_size=None, task=i+1, 
            allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
        ) for i in range(args.tasks)]
    else:
        precs = [evaluate.validate(
            model, test_datasets[i], verbose=False, test_size=None, task=i+1, 
            allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
        ) for i in range(args.tasks)]
        print("Precs: ", precs)
    average_precs = sum(precs) / args.tasks

    # -print on screen
    if verbose:
        print("\n Precision on test-set:")
        for i in range(args.tasks):
            print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
        print('=> Average precision over all {} tasks: {:.4f}\n'.format(args.tasks, average_precs))

    if verbose and args.time:
        print("=> Total training time = {:.1f} seconds\n".format(training_time))
    print("RAM AT THE END:", ramu.compute("AT THE END"))
    print("CPU AT THE END:", cpuu.compute("AT THE END"))


if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -run experiment
    run(args, verbose=True)
