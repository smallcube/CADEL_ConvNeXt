import os
import pickle
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from logger import Logger
import time
import numpy as np
from models.utils import *
from models.loss import *
from data_loader.utils import *
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from engine3 import train_one_epoch
from optim_factory import create_optimizer, LayerDecayValueAssigner
from timm.utils import ModelEma


class model():
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.training_opt = self.config['training_opt']
        self.dataset_params = config['dataset']
        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])
        #print("rank=",args.local_rank)
        self.rank = args.local_rank
        # Initialize model
        networks_args = config['networks']
        def_file = networks_args['def_file']
        model_args = networks_args['params']
        self.model = source_import(def_file).create_model(**model_args)

        torch.cuda.set_device(self.rank)
        dist.init_process_group(backend='nccl', rank=self.rank)
        self.device = torch.device('cuda', self.rank)
        self.data = get_dataloader(distributed=True, options=self.dataset_params)

        
        self.model = self.model.to(self.device)

        
        self.model_ema = None
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEma(
                self.model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
            print("Using EMA with decay = %.8f" % args.model_ema_decay)

        self.model_without_ddp = self.model
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        #print("Model = %s" % str(self.model_without_ddp))
        print('number of params:', n_parameters)

        total_batch_size = self.dataset_params['batch_size'] * args.update_freq #* utils.get_world_size()
        
        self.num_training_steps_per_epoch = len(self.data['train']) // args.update_freq
        print("total_batch_size=", total_batch_size, "   num_training_steps_per_epoch=", self.num_training_steps_per_epoch, "  len=", len(self.data['train']))
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % args.update_freq)
        print("Number of training examples = %d" % len(self.data['train']))
        print("Number of training training per epoch = %d" % self.num_training_steps_per_epoch)

        if args.layer_decay < 1.0 or args.layer_decay > 1.0:
            num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
            assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
                "Layer Decay impl only supports convnext_small/base/large/xlarge"
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        
        #self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], find_unused_parameters=False)
        self.model = nn.parallel.DistributedDataParallel(self.model, 
                                                            broadcast_buffers=True,
                                                            device_ids=[self.rank],
                                                            output_device=self.rank,
                                                            find_unused_parameters=True)
        self.model_without_ddp = self.model.module

        if 'model_dir' in self.config and self.config['model_dir'] is not None:
            self.load_model(self.config['model_dir'])

        #prepare optimizer:
        self.optimizer = create_optimizer(
            args, self.model_without_ddp, skip_list=None,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)

        self.loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

        print("Use Cosine LR scheduler")
        self.lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, self.training_opt['num_epochs'], self.num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )

        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        self.wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, self.num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(self.wd_schedule_values), min(self.wd_schedule_values)))

        utils.auto_load_model(
            args=args, model=self.model, model_without_ddp=self.model_without_ddp,
            optimizer=self.optimizer, loss_scaler=self.loss_scaler, model_ema=self.model_ema)

        max_accuracy = 0.0
        if args.model_ema and args.model_ema_eval:
            max_accuracy_ema = 0.0

        self.init_weight()
        self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
        self.logger.log_cfg(self.config)

    def init_weight(self):
        #prepare for logits adjustment
        cls_num = [0] * self.config['dataset']['num_classes']
        for label in self.data['train'].dataset.targets:
            cls_num[label] += 1
       
        cls_num = torch.tensor(cls_num).view(1, -1).cuda()
        self.distribution_source = torch.log(cls_num / torch.sum(cls_num)).view(1, -1)
        self.distribution_target = np.log(1.0 / self.config['dataset']['num_classes'])
    

    def train(self):
        # When training the network
        time.sleep(0.25)
        best_acc = 0.0
        best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        best_model_weights = copy.deepcopy(self.model.state_dict())
        self.reset_model(best_model_weights)
        total_steps = 0
        for epoch in range(0, end_epoch):
            total_steps += self.num_training_steps_per_epoch
            self.epoch = epoch
            torch.cuda.empty_cache()
            self.model.train()

            # Iterate over dataset
            total_preds, total_labels, loss_value = train_one_epoch(model=self.model, start_steps=epoch*self.num_training_steps_per_epoch, 
                                                        training_opt=self.training_opt,
                                                        data_loader=self.data['train'], 
                                                        optimizer=self.optimizer,
                                                        device=self.device, 
                                                        epoch=epoch, 
                                                        loss_scaler=self.loss_scaler, 
                                                        max_norm = self.args.clip_grad,
                                                        model_ema = self.model_ema, 
                                                        lr_schedule_values=self.lr_schedule_values, 
                                                        wd_schedule_values=self.wd_schedule_values,
                                                        num_training_steps_per_epoch=self.num_training_steps_per_epoch, 
                                                        update_freq=self.args.update_freq, 
                                                        use_amp=self.args.use_amp, 
                                                        mixup_param=self.training_opt['mixup'])

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)
            
            lr_current = max([param_group['lr'] for param_group in self.optimizer.param_groups])

            print_str = ['Epoch: [%d/%d]'
                            % (epoch+1, self.training_opt['num_epochs']),
                            'Step: %5d'
                            % (total_steps),
                            'Loss: %.4f'
                            % (loss_value),
                            'current_learning_rate: %0.5f'
                            % (lr_current)]
            print_write(print_str, self.log_file)

            
            # Log results
            self.logger.log_acc(rsls)

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                self.save_model(epoch, best_epoch, best_acc)
                best_model_weights = copy.deepcopy(self.model.state_dict())
                
            print('===> Saving checkpoint')
            self.save_latest(epoch)

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            #self.scheduler.step()
            #if self.criterion_optimizer:
            #    self.criterion_optimizer_scheduler.step()

            del self.logits
            torch.cuda.empty_cache()

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        #self.save_latest(epoch, best_epoch, best_acc)

        self.reset_model(best_model_weights)
        # Test on the test set
        self.eval('test' if 'test' in self.data else 'val')
        print('Done')

    def eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)

        # Calculate normal prediction accuracy
        rsl = {'train_all': 0., 'train_many': 0., 'train_median': 0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = \
                list(map(np.concatenate, [mixup_preds * 2, mixup_labels1 + mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1 - mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = ['\n Training acc Top1: %.3f \n' % (rsl['train_all']),
                     'Many_top1: %.3f' % (rsl['train_many']),
                     'Median_top1: %.3f' % (rsl['train_median']),
                     'Low_top1: %.3f' % (rsl['train_low']),
                     '\n']
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase='val', tao=1.0, post_hoc=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        
        torch.cuda.empty_cache()

        self.model.eval()

        self.total_logits = torch.empty((0, self.dataset_params['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        # Iterate over dataset
        for data in tqdm(self.data[phase]):
            inputs, labels = data[0].cuda(), data[1].cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                self.outputs = self.model(inputs)
                logits_backup = 0
                for i in range(0, len(self.outputs)):
                    if post_hoc:
                        logits_backup = logits_backup + self.outputs[i] + tao * (self.distribution_target - self.distribution_source)
                    else:
                        logits_backup = logits_backup + self.outputs[i]
                logits_backup /= len(self.outputs) - 1
                self.logits = logits_backup

                
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))

        
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1 = mic_acc_cal(preds[self.total_labels != -1],
                                             self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=False,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds[self.total_labels != -1],
                                 self.total_labels[self.total_labels != -1],
                                 self.data['train'],
                                 acc_per_cls=True)
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s'
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f'
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f'
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f'
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f'
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f'
                     % (self.low_acc_top1),
                     '\n']

        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}

        if phase == 'val':
            print_write(print_str, self.log_file)
        else:
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)

        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.cls_accs, f)
        return rsl

    
    
    def save_latest(self, epoch):
        model_states = {
            'epoch': epoch,
            'state_dict': self.model.state_dict()
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 'latest_model_checkpoint_%d.pth' % (self.rank))
        torch.save(model_states, model_dir)

    def save_model(self, epoch, best_epoch, best_acc):
        model_states = {'epoch': epoch,
                        'best_epoch': best_epoch,
                        'state_dict': self.model.state_dict(),
                        'best_acc': best_acc}

        model_dir = os.path.join(self.training_opt['log_dir'], 'final_model_checkpoint_%d.pth'%(self.rank))

        torch.save(model_states, model_dir)
    
    def reset_model(self, model_state):
        self.model.load_state_dict(model_state)

    def load_model(self, model_dir=None):
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')

        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))

        checkpoint = torch.load(model_dir, map_location='cpu')
        model_state = checkpoint['state_dict']
        self.model.load_state_dict(model_state)

    