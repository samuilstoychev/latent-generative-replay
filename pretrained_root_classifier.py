import torch
from torch.nn import functional as F
import torch.nn as nn
from continual_learner import ContinualLearner
from replayer import Replayer
import utils
import torchvision.models as models
from torchsummary import summary

class PretrainedRootClassifier(ContinualLearner, Replayer):

    def __init__(self, classes, latent_space, binaryCE=False, binaryCE_distill=False, AGEM=False, 
                 out_channels=5, kernel_size=5, root_type="VGG-16"):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.latent_space = latent_space
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.root_type = root_type

        # settings for training
        self.binaryCE = binaryCE                 #-> use binary (instead of multiclass) prediction error
        self.binaryCE_distill = binaryCE_distill #-> for classes from previous tasks, use the by the previous model
                                                 #   predicted probs as binary targets (only in Class-IL with binaryCE)
        self.AGEM = AGEM  #-> use gradient of replayed data as inequality constraint for (instead of adding it to)
                          #   the gradient of the current data (as in A-GEM, see Chaudry et al., 2019; ICLR)

        ######------SPECIFY MODEL------######
        # From https://github.com/pytorch/examples/blob/master/mnist/main.py
        
        if self.root_type == "VGG-16": 
            self.root = models.vgg16(pretrained=True)
        elif self.root_type == "MOBILENET-V2": 
            # self.root = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            self.root = models.mobilenet_v2(pretrained=True)
        elif self.root_type == "RESNET-18":
            resnet18 = models.resnet18(pretrained=True)
            self.root = torch.nn.Sequential(*list(resnet18.children())[:-1])
        elif self.root_type == "ALEXNET": 
            # self.root = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
            self.root = models.alexnet(pretrained=True)
        # Freeze root's parameters
        for param in self.root.parameters():
            param.requires_grad = False
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.latent_space, 128)
        self.fc2 = nn.Linear(128, classes)

    def list_init_layers(self):
        '''Return  list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.root
        list += self.dropout2
        list += self.fc1
        list += self.fc2
        return list

    @property
    def name(self):
        return "{}_c{}".format(self.root_type, self.classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classify(x)

    def feature_extractor(self, x):
        if self.root_type == "VGG-16": 
            x = self.root.features(x)
            x = self.root.avgpool(x)
            x = torch.flatten(x, 1)
            # Reduce space to 4096
            x = self.root.classifier[0](x)
            x = torch.sigmoid(x)
            return x
        elif self.root_type == "MOBILENET-V2":
            x = self.root.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            # Reduce space to 1280
            x = torch.sigmoid(x)
            return x
        elif self.root_type == "RESNET-18": 
            x = self.root(x)
            x = torch.flatten(x, 1)
            # Reduce space to 512
            x = torch.sigmoid(x)
            return x
        elif self.root_type == "ALEXNET":
            x = self.root.features(x)
            x = self.root.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.root.classifier[0](x)
            x = self.root.classifier[1](x)
            # Reduce space to 4096
            x = torch.sigmoid(x)
            return x

    def classify(self, x): 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def train_a_batch(self, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, task=1):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Should gradient be computed separately for each task? (needed when a task-mask is combined with replay)
        gradient_per_task = True if ((self.mask_dict is not None) and (x_ is not None)) else False


        ##--(1)-- REPLAYED DATA --##

        if x_ is not None:
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
            TaskIL = (type(y_)==list) if (y_ is not None) else (type(scores_)==list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            predL_r = [None]*n_replays
            distilL_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            if (not type(x_)==list) and (self.mask_dict is None):
                y_hat_all = self(x_)

            # Loop to evalute predictions on replay according to each previous task
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_)==list) or (self.mask_dict is not None):
                    x_temp_ = x_[replay_id] if type(x_)==list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id+1)
                    y_hat_all = self(x_temp_)

                # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in replayed task
                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    if self.binaryCE:
                        binary_targets_ = utils.to_one_hot(y_[replay_id].cpu(), y_hat.size(1)).to(y_[replay_id].device)
                        predL_r[replay_id] = F.binary_cross_entropy_with_logits(
                            input=y_hat, target=binary_targets_, reduction='none'
                        ).sum(dim=1).mean()     #--> sum over classes, then average over batch
                    else:
                        predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                    n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
                    kd_fn = utils.loss_fn_kd_binary if self.binaryCE else utils.loss_fn_kd
                    distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                                 target_scores=scores_[replay_id], T=self.KD_temp)
                # Weigh losses
                if self.replay_targets=="hard":
                    loss_replay[replay_id] = predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] = distilL_r[replay_id]

                # If needed, perform backward pass before next task-mask (gradients of all tasks will be accumulated)
                if gradient_per_task:
                    weight = 1 if self.AGEM else (1 - rnt)
                    weighted_replay_loss_this_task = weight * loss_replay[replay_id] / n_replays
                    weighted_replay_loss_this_task.backward()

        # Calculate total replay loss
        loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays

        # If using A-GEM, calculate and store averaged gradient of replayed data
        if self.AGEM and x_ is not None:
            # Perform backward pass to calculate gradient of replayed batch (if not yet done)
            if not gradient_per_task:
                loss_replay.backward()
            # Reorganize the gradient of the replayed batch as a single vector
            grad_rep = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_rep.append(p.grad.view(-1))
            grad_rep = torch.cat(grad_rep)
            # Reset gradients (with A-GEM, gradients of replayed batch should only be used as inequality constraint)
            self.optimizer.zero_grad()


        ##--(2)-- CURRENT DATA --##

        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            if self.binaryCE:
                # -binary prediction loss
                binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                if self.binaryCE_distill and (scores is not None):
                    classes_per_task = int(y_hat.size(1) / task)
                    binary_targets = binary_targets[:, -(classes_per_task):]
                    binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
                predL = None if y is None else F.binary_cross_entropy_with_logits(
                    input=y_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()     #--> sum over classes, then average over batch
            else:
                # -multiclass prediction loss
                predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

            # Weigh losses
            loss_cur = predL

            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # If backward passes are performed per task (e.g., XdG combined with replay), perform backward pass
            if gradient_per_task:
                weighted_current_loss = rnt*loss_cur
                weighted_current_loss.backward()
        else:
            precision = predL = None
            # -> it's possible there is only "replay" [e.g., for offline with task-incremental learning]


        # Combine loss from current and replayed batch
        if x_ is None or self.AGEM:
            loss_total = loss_cur
        else:
            loss_total = loss_replay if (x is None) else rnt*loss_cur+(1-rnt)*loss_replay


        ##--(3)-- ALLOCATION LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c>0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda>0:
            loss_total += self.ewc_lambda * ewc_loss


        # Backpropagate errors (if not yet done)
        if not gradient_per_task:
            loss_total.backward()

        # If using A-GEM, potentially change gradient:
        if self.AGEM and x_ is not None:
            # -reorganize gradient (of current batch) as single vector
            grad_cur = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur*grad_rep).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_rep*grad_rep).sum()
                grad_proj = grad_cur-(angle/length_rep)*grad_rep
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                        index += n_param

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item()/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item()/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'ewc': ewc_loss.item(), 'si_loss': surrogate_loss.item(),
            'precision': precision if precision is not None else 0.,
        }
