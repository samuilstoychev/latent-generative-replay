import torch
from torch import optim
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
import utils
from data import SubDataset, ExemplarDataset
from continual_learner import ContinualLearner
from cl_metrics import RAMU
from naive_rehearsal import ReplayBuffer
import evaluate

ramu = RAMU()

def train_cl(model, train_datasets, replay_mode="none", scenario="class",classes_per_task=None,iters=2000,batch_size=32,
             generator=None, gen_iters=0, use_exemplars=False, add_exemplars=False, buffer_size=1000, valid_datasets=None, early_stop=False, validation=False, 
             gen_loss_cbs=list(), loss_cbs=list()):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSeet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)'''

    peak_ramu = ramu.compute("TRAINING")
    valid_precs = []
    train_precs = []
    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    if replay_mode == "naive-rehearsal": 
        replay_buffer = ReplayBuffer(size=buffer_size, scenario=scenario)

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        prev_prec = 0.0
        peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
    

        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            active_classes = list(range(classes_per_task * task))

        # Reset state of optimizer(s) for every task (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if scenario=="task":
            up_to_task = task if replay_mode=="offline" else task-1
            iters_left_previous = [1]*up_to_task
            data_loader_previous = [None]*up_to_task

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))

        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
        for batch_index in range(1, iters_to_use+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
                # NOTE:  [train_dataset]  is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(data_loader)
            if Exact:
                if scenario=="task":
                    up_to_task = task if replay_mode=="offline" else task-1
                    batch_size_replay = int(np.ceil(batch_size/up_to_task)) if (up_to_task>1) else batch_size
                    # -in Task-IL scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[task_id]))
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id]==0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                train_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous==0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)


            # -----------------Collect data------------------#

            peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
            #####-----CURRENT BATCH-----#####
            if replay_mode=="offline" and scenario=="task":
                x = y = scores = None
            else:
                x, y = next(data_loader)                                    #--> sample training data of current task
                y = y-classes_per_task*(task-1) if scenario=="task" else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                """To PLot latent representations"""
                if batch_index == iters_to_use:
                    np.savetxt('R_x_' + str(task) + '.txt', x.reshape(x.shape[0], -1).cpu().data.numpy())
                    np.savetxt('R_x_labels_' + str(task) + '.txt', y.cpu().data.numpy())
                # If --bce, --bce-distill & scenario=="class", calculate scores of current batch with previous model
                binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                if binary_distillation and scenario=="class" and (previous_model is not None):
                    with torch.no_grad():
                        scores = previous_model(x)[:, :(classes_per_task * (task - 1))]
                else:
                    scores = None
            peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))

            #####-----REPLAYED BATCH-----#####
            if not Exact and not Generative and not Current:
                x_ = y_ = scores_ = None   #-> if no replay

            ##-->> Exact Replay <<--##
            if Exact:
                scores_ = None
                if scenario in ("domain", "class"):
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (model.replay_targets=="hard") else None
                    # If required, get target scores (i.e, [scores_]         -- using previous model, with no_grad()
                    if (model.replay_targets=="soft"):
                        with torch.no_grad():
                            scores_ = previous_model(x_)
                        scores_ = scores_[:, :(classes_per_task*(task-1))] if scenario=="class" else scores_
                        #-> when scenario=="class", zero probabilities will be added in the [utils.loss_fn_kd]-function
                elif scenario=="task":
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode=="offline" else task-1
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        x_.append(x_temp.to(device))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if model.replay_targets=="hard":
                            y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                    # If required, get target scores (i.e, [scores_]         -- using previous model
                    if (model.replay_targets=="soft") and (previous_model is not None):
                        scores_ = list()
                        for task_id in range(up_to_task):
                            with torch.no_grad():
                                scores_temp = previous_model(x_[task_id])
                            scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                            scores_.append(scores_temp)

            ##-->> Generative / Current Replay <<--##
            if Generative or Current:
                # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                x_ = x if Current else previous_generator.sample(batch_size)
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None):
                    with torch.no_grad():
                        all_scores_ = previous_model(x_)
                # -depending on chosen scenario, collect relevant predicted scores (per task, if required)
                if scenario in ("domain", "class") and (
                        (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None)
                ):
                    scores_ = all_scores_[:,:(classes_per_task * (task - 1))] if scenario == "class" else all_scores_
                    _, y_ = torch.max(scores_, dim=1)
                else:
                    # NOTE: it's possible to have scenario=domain with task-mask (so actually it's the Task-IL scenario)
                    # -[x_] needs to be evaluated according to each previous task, so make list with entry per task
                    scores_ = list()
                    y_ = list()
                    for task_id in range(task - 1):
                        # -if there is a task-mask (i.e., XdG is used), obtain predicted scores for each task separately
                        if hasattr(previous_model, "mask_dict") and previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(task=task_id + 1)
                            with torch.no_grad():
                                all_scores_ = previous_model(x_)
                        if scenario=="domain":
                            temp_scores_ = all_scores_
                        else:
                            temp_scores_ = all_scores_[:,
                                           (classes_per_task * task_id):(classes_per_task * (task_id + 1))]
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        scores_.append(temp_scores_)
                        y_.append(temp_y_)

                # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))

            # Validation for early stopping 
            if early_stop and valid_datasets and (batch_index % 100 == 0): 
                prec = evaluate.validate(
                    model, valid_datasets[task-1], verbose=False, test_size=None, task=task, 
                    allowed_classes=list(range(classes_per_task*(task-1), classes_per_task*(task))) if scenario=="task" else list(range(task))
                ) 
                if prec < prev_prec: 
                    prev_prec = 0.0
                    break 
                prev_prec = prec 

            #---> Train MAIN MODEL
            if batch_index <= iters:
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
                if replay_mode == "naive-rehearsal": 
                    replayed_data = replay_buffer.replay(batch_size)
                    if replayed_data: 
                        x_, y_ = zip(*replayed_data)
                        x_, y_ = torch.stack(x_), torch.tensor(y_)
                        x_ = x_.to(device) 
                        y_ = y_.to(device) 
                        if scenario == "task": 
                            y_ = [y_]
                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt = 1./task)
                if replay_mode == "naive-rehearsal": 
                    replay_buffer.add(zip(x, y))
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))

                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)

            #---> Train GENERATOR
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
                loss_dict = generator.train_a_batch(x, y, x_=x_, y_=y_, scores_=scores_, active_classes=active_classes,
                                                    task=task, rnt=1./task)
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))

                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)

        if validation and valid_datasets: 
            v_precs = [evaluate.validate(
                model, valid_datasets[i-1], verbose=False, test_size=None, task=i, 
                allowed_classes=list(range(classes_per_task*(i-1), classes_per_task*(i))) if scenario=="task" else list(range(task))
            ) for i in range(1, task+1)]
            t_precs = [evaluate.validate(
                model, train_datasets[i-1], verbose=False, test_size=None, task=i, 
                allowed_classes=list(range(classes_per_task*(i-1), classes_per_task*(i))) if scenario=="task" else list(range(task))
            ) for i in range(1, task+1)]
            valid_precs.append((task, batch_index, v_precs))
            train_precs.append((task, batch_index, t_precs))

        ##----------> UPON FINISHING EACH TASK...
        if replay_mode == "naive-rehearsal": 
            replay_buffer.update()

        # Close progres-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_mode == 'generative':
            Generative = True
            previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
            # if plot and task == 1:
            R_x_prime = previous_generator.sample(batch_size)
            all_scores_ = previous_model(R_x_prime)
            # temp_scores_ = all_scores_[:,
            #                (classes_per_task * task):(classes_per_task * (task + 1))]
            _, temp_y_ = torch.max(all_scores_, dim=1)

            np.savetxt('R_x_prime_' + str(task) + '.txt', R_x_prime.reshape(x.shape[0], -1).cpu().data.numpy())
            np.savetxt('R_x_prime_labels_' + str(task) + '.txt', temp_y_.cpu().data.numpy())
        elif replay_mode == 'current':
            Current = True
        elif replay_mode in ('exemplars', 'exact'):
            Exact = True
            if replay_mode == "exact":
                previous_datasets = train_datasets[:task]
            else:
                if scenario == "task":
                    previous_datasets = []
                    for task_id in range(task):
                        previous_datasets.append(
                            ExemplarDataset(
                                model.exemplar_sets[
                                (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                target_transform=lambda y, x=classes_per_task * task_id: y + x)
                        )
                else:
                    target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                    previous_datasets = [
                        ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]
        peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
    peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
    print("PEAK TRAINING RAM:", peak_ramu)
    if validation: 
        return (valid_precs, train_precs)
    return None

def pretrain_root(root_model, model, pretrain_dataset, batch_iterations=1000, batch_size=32, n_classes=10): 

    model.train()
    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    iters_left = 1

    active_classes = list(range(n_classes))
    n_loads = 0

    for batch_index in range(1, batch_iterations+1):
        iters_left -= 1
        if iters_left==0:
            data_loader = iter(utils.get_data_loader(pretrain_dataset, batch_size, cuda=cuda, drop_last=True))
            iters_left = len(data_loader)

        x, y = next(data_loader)                                #--> sample training data of current task
        n_loads += 1

        x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
        model.train_a_batch(x, y, active_classes=active_classes)

    # Transfer weights to root 
    model_params = model.named_parameters()
    root_params = root_model.named_parameters()
    root_params_dict = dict(root_params)

    for name, param in model_params: 
        if name in root_params_dict: 
            root_params_dict[name].data.copy_(param.data)

def pretrain_baseline(model, pretrain_dataset, batch_iterations=1000, batch_size=32, n_classes=10): 
    model.train()
    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # root_layers = model.get_root_layers()
    # top_layers_init = model.get_top_layers_init()
    root_model = model.get_sample_root()
    top_model = model.get_sample_top()

    root_params = root_model.named_parameters()
    top_params = top_model.named_parameters()
    model_params_dict = dict(model.named_parameters())

    # Note that we pre-train the model here using the `mnist_pretrain` slice that we have defined earlier. 
    iters_left = 1

    active_classes = list(range(n_classes))
    n_loads = 0 

    for batch_index in range(1, batch_iterations+1):
        iters_left -= 1
        if iters_left==0:
            data_loader = iter(utils.get_data_loader(pretrain_dataset, batch_size, cuda=cuda, drop_last=True))
            iters_left = len(data_loader)
            
        x, y = next(data_loader)                                #--> sample training data of current task
        n_loads += 1
         
        x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
        model.train_a_batch(x, y, active_classes=active_classes)

    # Freeze root layers 
    # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
    for name, param in root_params: 
        model_params_dict[name].requires_grad = False

    for name, param in top_params: 
        model_params_dict[name].data.copy_(param.data)
