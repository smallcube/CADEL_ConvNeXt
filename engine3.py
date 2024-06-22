import math
from typing import Iterable, Optional
import torch
from timm.utils import accuracy, ModelEma
from models.loss import *
from models.utils import *
from tqdm import tqdm

import utils

def train_one_epoch(model: torch.nn.Module, start_steps: int, training_opt,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, mixup_param=None):
    
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    total_preds, total_labels = [], []
    optimizer.zero_grad()
    #data_iter_step = 0
    total_loss = 0
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = data[0].to(device, non_blocking=True)
        targets = data[1].to(device, non_blocking=True)
        lam=1.0
        y_b = None

        if mixup_param is not None:
            samples, targets, y_b, lam = mixup_data(samples, targets, alpha=mixup_param['alpha'])
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                weight1, weight2 = None, None
                loss = 0
                logits_backup = 0
                for i in range(len(outputs)):
                    loss_i, weight1, weight2 = ensemble_loss(pred=outputs[i], target=targets, target2=y_b, lam=lam,
                                                            weight1=weight1, weight2=weight2,
                                                            bins=training_opt['bins'], 
                                                            gamma=training_opt['gamma'],
                                                            base_weight=training_opt['base_weight'])
                    loss = loss + loss_i
                    logits_backup = logits_backup + outputs[i]
                del weight1, weight2

        else: # full precision
            outputs = model(samples)
            weight1, weight2 = None, None
            loss = 0
            logits_backup = 0
            for i in range(len(outputs)):
                loss_i, weight1, weight2 = ensemble_loss(pred=outputs[i], target=targets, target2=y_b, lam=lam,
                                                        weight1=weight1, weight2=weight2,
                                                        bins=training_opt['bins'], 
                                                        gamma=training_opt['gamma'],
                                                        base_weight=training_opt['base_weight'])
                loss = loss + loss_i
                logits_backup = logits_backup + outputs[i]
            del weight1, weight2

        loss_value = loss.item()
        total_loss += loss_value

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        _, preds = torch.max(logits_backup, dim=-1)
        total_preds.append(torch2numpy(preds))
        total_labels.append(torch2numpy(targets))

        class_acc = (logits_backup.max(-1)[-1] == targets).float().mean()
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    total_loss /= update_freq*num_training_steps_per_epoch
    return total_preds, total_labels, total_loss

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
