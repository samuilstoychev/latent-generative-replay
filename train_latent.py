import torch
from torch import optim
import numpy as np
import tqdm
import copy
import utils
from cl_metrics import RAMU
from naive_rehearsal import ReplayBuffer
import evaluate_latent

ramu = RAMU()

def train_cl_latent(model, train_datasets, root=None, replay_mode="none", scenario="class",classes_per_task=None,iters=2000,batch_size=32,
             generator=None, gen_iters=0, metric_cbs=list(), buffer_size=1000, valid_datasets=None, early_stop=False, validation=False,
             gen_loss_cbs=list(), loss_cbs=list()):
    
    peak_ramu = ramu.compute("TRAINING")
    valid_precs = []
    train_precs = [] 
    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    # NOTE: Those correspond to exact replay, generative replay and current replay
    Generative = False
    previous_model = None

    # NOTE: We initalise `previous_generator` DURING the first iteration and AFTER the first reference
    # Loop over all tasks.
    # NOTE: 1 means 'start indexing from 1'. So task goes from 1 to N. 
    # NOTE: This is TASK_LOOP - iterating over the different TASKS

    if replay_mode == "naive-rehearsal": 
        replay_buffer = ReplayBuffer(size=buffer_size, scenario=scenario)

    for task, train_dataset in enumerate(train_datasets, 1):
        prev_prec = 0.0
        peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
        
        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = []
            for i in range(task):
                active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            active_classes = list(range(classes_per_task * task))
        # NOTE: for the Class-IL with MNIST --> 5 tasks each with 2 classes = 10 'active' classes in total. 

        # Reset state of optimizer(s) for every task (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = 1

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator: 
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))

        # Loop over all iterations
        # NOTE: Number of iterations specified in the function arguments. 
        batch_iterations = max(iters, gen_iters)
        peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))

        # NOTE: This is the second loop (BATCH_LOOP) - here we iterate over BATCHES.

    
        for batch_index in range(1, batch_iterations+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left == 0:
                data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)
    
                # -----------------Collect data------------------#

            peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
            # NOTE: x and y are the training data from the CURRENT task
            x, y = next(data_loader)  # --> sample training data of current task
            y = y - classes_per_task * (
                        task - 1) if scenario == "task" else y  # --> ITL: adjust y-targets to 'active range'
            x, y = x.to(device), y.to(device)  # --> transfer them to correct device
            
            """To PLot latent representations"""
            Rx = root(x)
            Rx = Rx.detach()
            Rx = Rx.to(device)
            peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))

            scores = None

            if not Generative:
                Rx_ = y_ = scores_ = None

            if Generative:
                # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                """To PLot latent representations"""
                Rx_ = previous_generator.sample(batch_size)
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None):
                    with torch.no_grad():
                        # NOTE: I guess this is the result (or the 'output') from the old 'solver'.
                        all_scores_ = previous_model(Rx_)
                # -depending on chosen scenario, collect relevant predicted scores (per task, if required)
                if scenario in ("domain", "class") and (
                        (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None)
                ):
                    # NOTE: Get the soft labels. Also notice that the number of classes is actually increasing
                    # with the number of tasks. For example, 1 and 2 in task 1 and then 1, 2, 3 and 4 in task 2.
                    scores_ = all_scores_[:, :(classes_per_task * (task - 1))] if scenario == "class" else all_scores_
                    # NOTE: And the hard labels
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
                                all_scores_ = previous_model(Rx_)
                        if scenario == "domain":
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
                prec = evaluate_latent.validate(
                    model, valid_datasets[task-1], root=root, verbose=False, test_size=None, task=task, 
                    allowed_classes=list(range(classes_per_task*(task-1), classes_per_task*(task))) if scenario=="task" else list(range(task))
                ) 
                if prec < prev_prec: 
                    prev_prec = 0.0
                    break 
                prev_prec = prec 

            #---> Train MAIN MODEL
            # NOTE: This will always hold as long as iters >= gen_iters 
            # Remember that batch_index goes up to max(iters, gen_iters). So the main model will no longer be trained 
            # if it has already been trained the required `iters` times. 
            if batch_index <= iters:
                # Train the main model with this batch
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
                if replay_mode == "naive-rehearsal": 
                    replayed_data = replay_buffer.replay(batch_size)
                    if replayed_data: 
                        Rx_, y_ = zip(*replayed_data)
                        Rx_, y_ = torch.stack(Rx_), torch.tensor(y_)
                        Rx_ = Rx_.to(device)
                        y_ = y_.to(device)
                        if scenario == "task": 
                            y_ = [y_] 
                loss_dict = model.train_a_batch(Rx, y, x_=Rx_, y_=y_, scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt = 1./task)
                if replay_mode == "naive-rehearsal": 
                    replay_buffer.add(zip(Rx, y))
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))

                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)

            #---> Train GENERATOR
            # NOTE: Again, remember that batch_index goes up to max(iters, gen_iters). 
            # So, we only train the generator as many times as we need (and not more). 
            if batch_index <= gen_iters and generator:

                # Train the generator with this batch
                # NOTE: I think (and I hope) that the generator does not actually need `y` to be passed. 
                # It seems like it's only there for precision? 
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
                loss_dict = generator.train_a_batch(Rx, y, x_=Rx_, y_=y_, scores_=scores_, active_classes=active_classes,
                                                    task=task, rnt=1./task)
                peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))

                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)


        
        if validation and valid_datasets:
            v_precs = [evaluate_latent.validate(
                model, valid_datasets[i-1], root=root, verbose=False, test_size=None, task=i, 
                allowed_classes=list(range(classes_per_task*(i-1), classes_per_task*(i))) if scenario=="task" else list(range(task))
            ) for i in range(1, task+1)]
            t_precs = [evaluate_latent.validate(
                model, train_datasets[i-1], root=root, verbose=False, test_size=None, task=i, 
                allowed_classes=list(range(classes_per_task*(i-1), classes_per_task*(i))) if scenario=="task" else list(range(task))
            ) for i in range(1, task+1)]
            valid_precs.append((task, batch_index, v_precs))
            train_precs.append((task, batch_index, t_precs))

        # NOTE: This bit is still within the TASK_LOOP. That is, it executes after each loop iteration (i.e. after
        # each task has completed). 
        ##----------> UPON FINISHING EACH TASK...
        if replay_mode == "naive-rehearsal": 
            replay_buffer.update()

        # Close progres-bar(s)
        progress.close()
        if generator: 
            progress_gen.close()

        # Calculate statistics required for metrics
        # NOTE: Those are statistics at a task (not batch) level. 
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task)

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        # NOTE: VERY IMPORTANT - see how both Generative and previous_generator are initiated AFTER the first cycle. 
        # That is because the first iteration is 'special' - there is no 'previous' model to remember. 
        if generator:
            Generative = True
            previous_generator = copy.deepcopy(generator).eval()
            # if plot and task == 1:
            R_x_prime = previous_generator.sample(batch_size)
            all_scores_ = previous_model(R_x_prime)
            # temp_scores_ = all_scores_[:,
            #                (classes_per_task * task):(classes_per_task * (task + 1))]
            _, temp_y_ = torch.max(all_scores_, dim=1)

            np.savetxt('R_x_prime_' + str(task) + '.txt', R_x_prime.cpu().data.numpy())
            np.savetxt('R_x_prime_labels_' + str(task) + '.txt', temp_y_.cpu().data.numpy())
                
        peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
        print("PEAK TRAINING RAM:", peak_ramu)
    if validation:
        return (valid_precs, train_precs)
    return None
