from socket import MsgFlag
import torch
import warnings 

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
        #'encoder': self.model.encoder.state_dict(),
        #'optimizer': optimizer.state_dict(),
    }
    if self.args.multi_task:
        if self.args.method == "HPS" or self.args.method == "MTAN":
            state["encoder"] = self.model.encoder.state_dict()
            for name, layer in self.model.named_children():
                if name == "encoder":
                    continue
                state[name] = self.model.decoder[name].state_dict()
        elif self.args.method == "HPS_v3":
            state["encoder"] = self.model.encoder.state_dict()
            for name, layer in self.model.named_children():
                if name == "encoder" or name == "discriminator" or name == "kl_loss":
                    continue
                state[name] = self.model.decoder[name].state_dict()
        
        elif self.args.method == "HPS_v4":
            state["encoder"] = self.model.encoder.state_dict()
            for name, layer in self.model.named_children():
                if name == "encoder" or name == "adver_loss" or name == "discriminator":
                    continue
                state[name] = self.model.decoder[name].state_dict()

        elif self.args.method == "Nova":
            state["encoder"] = self.model.encoder.state_dict()
            state["seg_head"] = self.model.seg_head.state_dict()
            state["seg_model"] = self.model.seg_model.state_dict()
            for name, layer in self.model.named_children():
                if name == "encoder" or name == "seg_head" or name == "seg_model":
                    continue
                state[name] = self.model.decoder[name].state_dict()

        elif self.args.method == "MMoE":
            state["encoder"] = self.model.encoder.state_dict()
            state["gate_specific"] = self.model.gate_specific.state_dict()
            for name, layer in self.model.named_children():
                if name == "encoder" or name == "gate_specific":
                    continue
                state[name] = self.model.decoder[name].state_dict()

        elif self.args.method == "DSelectK":
            state["encoder"] = self.model.encoder.state_dict()
            state["_z_logits"] = self.model._z_logits.state_dict()
            state["_w_logits"] = self.model._w_logits.state_dict()
            for name, layer in self.model.named_children():
                if name == "encoder" or name == "gate_specific" or name == "_z_logits" or name == "_w_logits":
                    continue
                state[name] = self.model.decoder[name].state_dict()
        
        elif self.args.method == "CGC":
            state["gate_specific"] = self.model.gate_specific.state_dict()
            state["experts_shared"] = self.model.experts_shared.state_dict()
            state["experts_specific"] = self.model.experts_specific.state_dict()
            for name, layer in self.model.named_children():
                if name == "experts_shared" or name == "gate_specific" or name == "experts_specific":
                    continue
                state[name] = self.model.decoder[name].state_dict()

        elif self.args.method == "Adapter":
            state["model"] = self.model.state_dict()
        
        else:
            terminal_msg(f"Wrong method: {self.args.method}", "F")

    else:
        state["encoder"] = self.model.encoder.state_dict()
        state["decoder"] = self.model.decoder.state_dict()

    if save_best:
        best_path = str('archive/checkpoints/' + arch + '/model_best.pth')
        torch.save(state, best_path)
        terminal_msg("Best model detected, saved ckpt to '{}'.".format(best_path), "C")
        return

    filename = str('archive/checkpoints/' + arch + '/epoch_{}.pth'.format(epoch))
    torch.save(state, filename)
    terminal_msg("Successfully saved ckpt to '{}'.".format(filename), "C")


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
        terminal_msg("Architecture in ckpt is different from the model ({}).".format(type(self.model).__name__), "F")
        exit()
    
    if self.args.multi_task:
        if self.args.method == "HPS" or self.args.method == "MTAN":
            self.model.encoder.load_state_dict(checkpoint['encoder'])
            for name, layer in self.model.named_children():
                if name == "encoder":
                    continue
                self.model.decoder[name].load_state_dict(checkpoint[name])
        
        elif self.args.method == "Nova":
            self.model.encoder.load_state_dict(checkpoint['encoder'])
            self.model.seg_head.load_state_dict(checkpoint['seg_head'])
            self.model.seg_model.load_state_dict(checkpoint['seg_model'])
            for name, layer in self.model.named_children():
                if name == "encoder" or name == "seg_head" or name == "seg_model":
                    continue
                self.model.decoder[name].load_state_dict(checkpoint[name])
        
        elif self.args.method == "MMoE":
            self.model.encoder.load_state_dict(checkpoint['encoder'])
            self.model.gate_specific.load_state_dict(checkpoint['gate_specific'])
            for name, layer in self.model.named_children():
                if name == "encoder" or name == "gate_specific":
                    continue
                self.model.decoder[name].load_state_dict(checkpoint[name])
        
        elif self.args.method == "CGC":
            self.model.experts_shared.load_state_dict(checkpoint['experts_shared'])
            self.model.experts_specific.load_state_dict(checkpoint['experts_specific'])
            self.model.gate_specific.load_state_dict(checkpoint['gate_specific'])
            for name, layer in self.model.named_children():
                if name == "experts_shared" or name == "gate_specific" or name == "experts_specific":
                    continue
                self.model.decoder[name].load_state_dict(checkpoint[name])
        elif self.args.method == "Adapter":
            self.model.load_state_dict(checkpoint['model'])
    else:
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.decoder.load_state_dict(checkpoint['decoder'])
    #self.optimizer.load_state_dict(checkpoint['optimizer'])
    terminal_msg("Checkpoint loaded successfully (epoch {})!".format(checkpoint['epoch']), "C")
