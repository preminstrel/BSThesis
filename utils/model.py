from socket import MsgFlag
import torch
from .info import terminal_msg


def count_parameters(model):
    """
    Count the number of parameters in a network.
    return params, trainable_params
    """
    num_params = 0
    num_trainable_params = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_trainable_params += p.numel()
    #print(f"Parameter number of {name}: {num_params / 1e6:.4f}M ({num_trainable_params / 1e6:.4f}M trainable)")
    return num_params, num_trainable_params


def save_checkpoint(self, epoch, save_best=False):
    """
    Saving checkpoints
    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(self.model).__name__
    terminal_msg("Saving ckpt of model '{}' ...".format(arch), "E")
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': self.model.resnet.state_dict(),
        'optimizer': self.model.optimizer.state_dict(),
    }
    filename = str('archive/checkpoints/' + arch + '/epoch_{}.pth'.format(epoch))
    torch.save(state, filename)
    terminal_msg("Successfully saved ckpt to '{}'.".format(filename), "C")

    if save_best:
        best_path = str('archive/checkpoints/' + arch + '/model_best.pth')
        torch.save(state, best_path)
        terminal_msg("Best model detected, saved ckpt to '{}'.".format(best_path), "C")


def resume_checkpoint(self, resume_path):
    """
    Resume from saved checkpoints
    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    terminal_msg("Loading checkpoint '{}' ...".format(resume_path), "E")
    checkpoint = torch.load(resume_path)
    self.start_epoch = checkpoint['epoch'] + 1

    # load architecture params from checkpoint.
    if checkpoint['arch'] != type(self.model).__name__:
        terminal_msg("Architecture in ckpt is different from the model.", "F")
        exit()
    self.model.resnet.load_state_dict(checkpoint['state_dict'])
    self.mode.optimizer.load_state_dict(checkpoint['optimizer'])
    terminal_msg("Checkpoint loaded successfully!", "C")
