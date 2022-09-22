from torch.optim import lr_scheduler
import sys
import torch
def get_lr_scheduler(lr_scheduler_name , optimizer, epochs, steps_per_epoch):
    if lr_scheduler_name == "StepLR":
        sched = lr_scheduler.StepLR(optimizer, step_size=7, gamma = 0.7, verbose= True)
    elif lr_scheduler_name == "OneCycleLR":
        sched = lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01 , epochs=epochs,
                                        steps_per_epoch=steps_per_epoch, verbose=True)
    elif lr_scheduler_name == "LambdaLR":
        #lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        sched = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2], verbose=True)
    elif lr_scheduler_name == "ExponentialLR":
        sched = lr_scheduler.ExponentialLR(optimizer, gamma = 0.5, verbose=True)
    else:
        print(f"{lr_scheduler} is not a valid learning-rate scheduler name.")
        sys.exit()
    return sched

def get_opt_func(optimizer, params, weight_decay):
    if optimizer == "Adadelta" :
        opt_func = torch.optim.Adadelta(params, weight_decay = weight_decay)
    elif optimizer == "Adagrad":
        opt_func = torch.optim.Adagrad(params, weight_decay = weight_decay)
    elif optimizer == "Adam":
        opt_func = torch.optim.Adam(params, weight_decay = weight_decay)
    elif optimizer == "RMSprop":
        opt_func = torch.optim.RMSprop(params, weight_decay = weight_decay)
    elif optimizer == "SGD":
        opt_func = torch.optim.SGD(params, lr=0.01, momentum=0.9)
    else :
        print(f"{optimizer} is not a valid optimizer name.")
    return opt_func
