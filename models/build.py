from ast import arg
from json import decoder
from .encoder_decoder import Encoder, Decoder_multi_classification, Decoder_single_classification
from .encoder_decoder import get_task_head, get_task_loss
import torch.nn as nn
import torch
import numpy as np
import math

from utils.info import terminal_msg, get_device
from utils.model import count_parameters
from models.resnet import resnet50

class build_single_task_model(nn.Module):
    '''
    build single-task model as baselines
    '''
    def __init__(self, args):
        super(build_single_task_model, self).__init__()
        self.encoder = Encoder()
        self.args = args
        if args.data == "ODIR-5K":
            self.decoder = Decoder_multi_classification(num_class = 8)
            type(self).__name__ = "ODIR-5K"
        elif args.data == "RFMiD":
            self.decoder = Decoder_multi_classification(num_class = 29)
            type(self).__name__ = "RFMiD"
        elif args.data == "TAOP":
            self.decoder = Decoder_single_classification(num_class = 5)
            type(self).__name__ = "TAOP"
        elif args.data == "APTOS":
            self.decoder = Decoder_single_classification(num_class = 5)
            type(self).__name__ = "APTOS"
        elif args.data == "Kaggle":
            self.decoder = Decoder_single_classification(num_class = 5)
            type(self).__name__ = "Kaggle"
        elif args.data == "DR+":
            self.decoder = Decoder_multi_classification(num_class = 28)
            type(self).__name__ = "DR+"
        elif args.data == "AMD":
            self.decoder = Decoder_single_classification(num_class = 2)
            type(self).__name__ = "AMD"
        elif args.data == "DDR":
            self.decoder = Decoder_single_classification(num_class = 6)
            type(self).__name__ = "DDR"
        elif args.data == "LAG":
            self.decoder = Decoder_single_classification(num_class = 2)
            type(self).__name__ = "LAG"
        elif args.data == "PALM":
            self.decoder = Decoder_single_classification(num_class = 2)
            type(self).__name__ = "PALM"
        elif args.data == "REFUGE":
            self.decoder = Decoder_single_classification(num_class = 2)
            type(self).__name__ = "REFUGE"
        else:
            terminal_msg("Args.Data Error (From build_single_task_model.__init__)", "F")
            exit()


        self.ODIR_5K_bce_loss = nn.BCELoss()

        self.RFMiD_bce_loss = nn.BCELoss()

        self.KaggleDR_bce_loss = nn.BCELoss()

        self.nll_loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=args.lr)

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

    def forward(self, img):
        return self.decoder(self.encoder(img))

    def process(self, img, gt):
        pred = self(img)
        self.optimizer.zero_grad()
        if self.args.data == "ODIR-5K":
            loss = self.ODIR_5K_bce_loss(pred, gt)
        elif self.args.data == "RFMiD":
            loss = self.RFMiD_bce_loss(pred, gt)
        elif self.args.data == "DR+":
            loss = self.KaggleDR_bce_loss(pred, gt)
        elif self.args.data in ["TAOP", "APTOS", "Kaggle", "AMD", "DDR", "LAG", "PALM", "REFUGE"]:
            if gt.shape[0] == 1:
                gt = gt[0].long()
            else:
                gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
            #print(pred.shape, gt.shape)
            loss = self.nll_loss(pred, gt)
            pred = torch.argmax(pred, dim = 1)
        else:
            terminal_msg("Error (From build_single_task_model.process)", "F")
            exit()
        return pred, loss

    def backward(self, loss = None):
        loss.backward()
        self.optimizer.step()

class build_hard_param_share_model(nn.Module):
    '''
    build hard params shared multi-task model
    '''
    def __init__(self, args):
        super(build_hard_param_share_model, self).__init__()
        self.encoder = Encoder()
        type(self).__name__ = "Hard_Params"
        self.args = args
        self.decoder = get_task_head(args.data)
        self.loss = get_task_loss(args.data)
        device = get_device()

        self.encoder.to(device)
        num_params, num_trainable_params = count_parameters(self.encoder)

        decoder_params = []
        for i in self.decoder:
            decoder_params += list(self.decoder[i].parameters())
            self.decoder[i].to(device)
            num_params_increment, num_trainable_params_increment = count_parameters(self.decoder[i])
            num_params += num_params_increment
            num_trainable_params += num_trainable_params_increment
        
        self.num_params = num_params
        self.num_trainable_params = num_trainable_params

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + decoder_params, lr=args.lr, betas=(0.5, 0.999))

        self.add_module("encoder", self.encoder)
        for i in self.decoder:
            self.add_module(str(i), self.decoder[i])

    def forward(self, img, head):
        presentation = self.encoder(img)
        pred = self.decoder[head](presentation)
        return presentation, pred

    def process(self, img, gt, head):
        presentation, pred = self(img, head)
        self.optimizer.zero_grad()

        if head in ["TAOP", "APTOS", "Kaggle", "AMD", "DDR", "LAG", "PALM", "REFUGE"]:
            if gt.shape[0] == 1:
                gt = gt[0].long()
            else:
                gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
            loss = self.loss[head](pred, gt)
            pred = torch.argmax(pred, dim = 1)
            return pred, loss

        loss = self.loss[head](pred, gt)

        return pred, loss

    def backward(self, loss = None):
        loss.backward()
        self.optimizer.step()

class build_MMoE_model(nn.Module):
    '''
    build MMoE multi-task model
    '''
    def __init__(self, args):
        super(build_MMoE_model, self).__init__()
        type(self).__name__ = "MMoE"
        self.args = args
        self.task_name = args.data.split(", ")
        self.rep_grad = True

        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}

        self.decoder = get_task_head(args.data)
        self.loss = get_task_loss(args.data)
        device = get_device()

        self.input_size = np.array(3*224*224, dtype=int).prod()
        self.num_experts = 3 # num of shared experts
        self.encoder = nn.ModuleList([resnet50(pretrained=True) for _ in range(self.num_experts)])
        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size, self.num_experts),
                                                                nn.Softmax(dim=-1)) for task in self.task_name})

        self.encoder.to(device)
        self.gate_specific.to(device)
        num_params, num_trainable_params = count_parameters(self.encoder)
        gate_num_params, gate_num_trainable_params = count_parameters(self.gate_specific)

        num_params = num_params + gate_num_params
        num_trainable_params = num_trainable_params + gate_num_trainable_params

        decoder_params = []
        gate_params = []
        for i in self.decoder:
            decoder_params += list(self.decoder[i].parameters())
            gate_params += list(self.gate_specific[i].parameters())
            self.decoder[i].to(device)
            num_params_increment, num_trainable_params_increment = count_parameters(self.decoder[i])
            num_params += num_params_increment
            num_trainable_params += num_trainable_params_increment
        
        self.num_params = num_params
        self.num_trainable_params = num_trainable_params

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + gate_params + decoder_params, lr=args.lr, betas=(0.5, 0.999))

        self.add_module("encoder", self.encoder)
        for i in self.decoder:
            self.add_module(str(i), self.decoder[i])

    def forward(self, img, head):
        experts_shared_rep = torch.stack([e(img) for e in self.encoder]) # [3, batch, 2048, 7, 7]
        selector = self.gate_specific[head](torch.flatten(img, start_dim=1)) # [batch, 3]
        gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector) # [batch, 2048, 7, 7]
        gate_rep = self._prepare_rep(gate_rep, head, same_rep=False) # [batch, 2048, 7, 7]
        pred = self.decoder[head](gate_rep) 
        return gate_rep, pred

    def process(self, img, gt, head):
        representation, pred = self(img, head)
        self.optimizer.zero_grad()

        if head in ["TAOP", "APTOS", "Kaggle", "AMD", "DDR", "LAG", "PALM", "REFUGE"]:
            if gt.shape[0] == 1:
                gt = gt[0].long()
            else:
                gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
            loss = self.loss[head](pred, gt)
            pred = torch.argmax(pred, dim = 1)
            return pred, loss

        loss = self.loss[head](pred, gt)

        return pred, loss

    def backward(self, loss = None):
        loss.backward()
        self.optimizer.step()

    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad()
        
    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep

class build_CGC_model(nn.Module):
    '''
    build CGC multi-task model
    '''
    def __init__(self, args):
        super(build_CGC_model, self).__init__()
        type(self).__name__ = "CGC"
        
        self.args = args
        self.task_name = args.data.split(", ")
        self.rep_grad = True

        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}

        self.decoder = get_task_head(args.data)
        self.loss = get_task_loss(args.data)
        device = get_device()
        self.input_size = np.array(3*224*224, dtype=int).prod()

        args_num_experts = [1 for i in range(len(self.task_name) + 1)] # experts = 1
        self.num_experts = {task: args_num_experts[tn+1] for tn, task in enumerate(self.task_name)}
        self.num_experts['share'] = args_num_experts[0]
        
        self.experts_shared = nn.ModuleList([resnet50(pretrained=True) for _ in range(self.num_experts['share'])])
        self.experts_specific = nn.ModuleDict({task: nn.ModuleList([resnet50(pretrained=True) for _ in range(self.num_experts[task])]) for task in self.task_name})
        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size, 
                                                                            self.num_experts['share']+self.num_experts[task]),
                                                                nn.Softmax(dim=-1)) for task in self.task_name})


        self.experts_specific.to(device)
        self.gate_specific.to(device)
        num_params, num_trainable_params = count_parameters(self.experts_specific)
        gate_num_params, gate_num_trainable_params = count_parameters(self.gate_specific)
        shared_num_params, shared_num_trainable_params = count_parameters(self.experts_shared)

        num_params = num_params + gate_num_params + shared_num_params
        num_trainable_params = num_trainable_params + gate_num_trainable_params + shared_num_trainable_params

        decoder_params = []
        gate_params = []
        for i in self.decoder:
            decoder_params += list(self.decoder[i].parameters())
            gate_params += list(self.gate_specific[i].parameters())
            self.decoder[i].to(device)
            num_params_increment, num_trainable_params_increment = count_parameters(self.decoder[i])
            num_params += num_params_increment
            num_trainable_params += num_trainable_params_increment
        
        self.num_params = num_params
        self.num_trainable_params = num_trainable_params

        self.optimizer = torch.optim.Adam(list(self.experts_specific.parameters()) + gate_params + decoder_params, lr=args.lr, betas=(0.5, 0.999))

        for i in self.decoder:
            self.add_module(str(i), self.decoder[i])

    def forward(self, img, head):
        experts_shared_rep = torch.stack([e(img) for e in self.experts_shared]) # [share_exp, batch, 2048, 7, 7]
        experts_specific_rep = torch.stack([e(img) for e in self.experts_specific[head]]) # [specific_exp, batch, 2048, 7, 7]
        selector = self.gate_specific[head](torch.flatten(img, start_dim=1)) # [batch, share + specific]
        gate_rep = torch.einsum('ij..., ji -> j...', torch.cat([experts_shared_rep, experts_specific_rep], dim=0), selector) # [batch, 2048, 7, 7]
        gate_rep = self._prepare_rep(gate_rep, head, same_rep=False) # [batch, 2048, 7, 7]
        pred = self.decoder[head](gate_rep) 
        return gate_rep, pred

    def process(self, img, gt, head):
        representation, pred = self(img, head)
        self.optimizer.zero_grad()

        if head in ["TAOP", "APTOS", "Kaggle", "AMD", "DDR", "LAG", "PALM"]:
            if gt.shape[0] == 1:
                gt = gt[0].long()
            else:
                gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
            loss = self.loss[head](pred, gt)
            pred = torch.argmax(pred, dim = 1)
            return pred, loss

        loss = self.loss[head](pred, gt)

        return pred, loss

    def backward(self, loss = None):
        loss.backward()
        self.optimizer.step()

    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad()
        
    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep

class build_MTAN_model(nn.Module):
    '''
    build MTAN multi-task model
    '''
    def __init__(self, args):
        super(build_MTAN_model, self).__init__()
        type(self).__name__ = "MTAN"
        
        self.args = args
        self.task_name = args.data.split(", ")
        self.rep_grad = True
        device = get_device()

        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}

        self.encoder = resnet50(pretrained=True)
        self.decoder = get_task_head(args.data)
        self.loss = get_task_loss(args.data)

        try: 
            callable(eval('self.encoder.layer1'))
            self.encoder = _transform_resnet_MTAN(self.encoder.to(device), self.task_name, device)
        except:
            self.encoder.resnet_network = _transform_resnet_MTAN(self.encoder.resnet_network.to(device), self.task_name, device)

        num_params, num_trainable_params = count_parameters(self.encoder)

        decoder_params = []
        for i in self.decoder:
            decoder_params += list(self.decoder[i].parameters())
            self.decoder[i].to(device)
            num_params_increment, num_trainable_params_increment = count_parameters(self.decoder[i])
            num_params += num_params_increment
            num_trainable_params += num_trainable_params_increment
        
        self.num_params = num_params
        self.num_trainable_params = num_trainable_params
        
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + decoder_params, lr=args.lr, betas=(0.5, 0.999))

        for i in self.decoder:
            self.add_module(str(i), self.decoder[i])

    def forward(self, img, head):
        try:
            callable(eval('self.encoder.resnet_network'))
            self.encoder.resnet_network.forward_task = head
        except:
            self.encoder.forward_task = head
        s_rep = self.encoder(img) # torch.Size([batch, 2048, 7, 7])
        for tn, task in enumerate(self.task_name):
            if self.task_name is not None and task != head:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep # torch.Size([batch, 2048, 7, 7])
            pred = self.decoder[task](ss_rep)

        return ss_rep, pred

    def process(self, img, gt, head):
        representation, pred = self(img, head)
        self.optimizer.zero_grad()

        if head in ["TAOP", "APTOS", "Kaggle", "AMD", "DDR", "LAG", "PALM"]:
            if gt.shape[0] == 1:
                gt = gt[0].long()
            else:
                gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
            loss = self.loss[head](pred, gt)
            pred = torch.argmax(pred, dim = 1)
            return pred, loss

        loss = self.loss[head](pred, gt)

        return pred, loss
    
    def backward(self, loss = None):
        loss.backward()
        self.optimizer.step()

class build_DSelectK_model(nn.Module):
    '''
    build DSelectK multi-task model
    '''
    def __init__(self, args):
        super(build_DSelectK_model, self).__init__()
        type(self).__name__ = "DSelectK"
        self.args = args
        self.task_name = args.data.split(", ")
        self.rep_grad = True

        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}

        self.decoder = get_task_head(args.data)
        self.loss = get_task_loss(args.data)
        device = get_device()

        self.input_size = np.array(3*224*224, dtype=int).prod()
        self.num_experts = 3 # num of shared experts
        self.encoder = nn.ModuleList([resnet50(pretrained=True) for _ in range(self.num_experts)])

        self._num_nonzeros = 1
        self._gamma = 1
        
        self._num_binary = math.ceil(math.log2(self.num_experts))
        self._power_of_2 = (self.num_experts == 2 ** self._num_binary)
        
        self._z_logits = nn.ModuleDict({task: nn.Linear(self.input_size, 
                                                        self._num_nonzeros*self._num_binary) for task in self.task_name})
        self._w_logits = nn.ModuleDict({task: nn.Linear(self.input_size, self._num_nonzeros) for task in self.task_name})

        # initialization
        for param in self._z_logits.parameters():
            param.data.uniform_(-self._gamma/100, self._gamma/100)
        for param in self._w_logits.parameters():
            param.data.uniform_(-0.05, 0.05)
        
        binary_matrix = np.array([list(np.binary_repr(val, width=self._num_binary)) \
                                  for val in range(self.num_experts)]).astype(bool)
        self._binary_codes = torch.from_numpy(binary_matrix).to(device).unsqueeze(0)  

        self.encoder.to(device)
        num_params, num_trainable_params = count_parameters(self.encoder)

        decoder_params = []

        for i in self.decoder:
            decoder_params += list(self.decoder[i].parameters())
            self.decoder[i].to(device)
            num_params_increment, num_trainable_params_increment = count_parameters(self.decoder[i])
            num_params += num_params_increment
            num_trainable_params += num_trainable_params_increment
        
        self.num_params = num_params
        self.num_trainable_params = num_trainable_params

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + decoder_params, lr=args.lr, betas=(0.5, 0.999))

        self.add_module("encoder", self.encoder)
        for i in self.decoder:
            self.add_module(str(i), self.decoder[i])
    
    def _smooth_step_fun(self, t, gamma=1.0):
        return torch.where(t<=-gamma/2, torch.zeros_like(t, device=t.device),
                   torch.where(t>=gamma/2, torch.ones_like(t, device=t.device),
                         (-2/(gamma**3))*(t**3) + (3/(2*gamma))*t + 1/2))
    
    def _entropy_reg_loss(self, inputs):
        loss = -(inputs*torch.log(inputs+1e-6)).sum() * 1e-6
        if not self._power_of_2:
            loss += (1/inputs.sum(-1)).sum()
        loss.backward(retain_graph=True)
    
    def forward(self, inputs, head=None):
        experts_shared_rep = torch.stack([e(inputs) for e in self.encoder])
        sample_logits = self._z_logits[head](torch.flatten(inputs, start_dim=1))
        sample_logits = sample_logits.reshape(-1, self._num_nonzeros, 1, self._num_binary)
        smooth_step_activations = self._smooth_step_fun(sample_logits)
        selector_outputs = torch.where(self._binary_codes.unsqueeze(0), smooth_step_activations, 
                                        1 - smooth_step_activations).prod(3)
        selector_weights = nn.functional.softmax(self._w_logits[head](torch.flatten(inputs, start_dim=1)), dim=1)
        expert_weights = torch.einsum('ij, ij... -> i...', selector_weights, selector_outputs)
        gate_rep = torch.einsum('ij, ji... -> i...', expert_weights, experts_shared_rep)
        gate_rep = self._prepare_rep(gate_rep, head, same_rep=False)
        out = self.decoder[head](gate_rep)

        if self.args.mode == 'train':
            self._entropy_reg_loss(selector_outputs)
        
        return gate_rep, out

    def process(self, img, gt, head):
        representation, pred = self(img, head)
        self.optimizer.zero_grad()

        if head in ["TAOP", "APTOS", "Kaggle", "AMD", "DDR", "LAG", "PALM", "REFUGE"]:
            if gt.shape[0] == 1:
                gt = gt[0].long()
            else:
                gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
            loss = self.loss[head](pred, gt)
            pred = torch.argmax(pred, dim = 1)
            return pred, loss

        loss = self.loss[head](pred, gt)

        return pred, loss

    def backward(self, loss = None):
        loss.backward()
        self.optimizer.step()

    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad()
        
    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep

class _transform_resnet_MTAN(nn.Module):
    def __init__(self, resnet_network, task_name, device):
        super(_transform_resnet_MTAN, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.forward_task = None
        
        self.expansion = 4 if resnet_network.feature_dim == 2048 else 1
        ch = np.array([64, 128, 256, 512]) * self.expansion
        self.shared_conv = nn.Sequential(resnet_network.conv1, resnet_network.bn1, 
                                         resnet_network.relu, resnet_network.maxpool)
        self.shared_layer, self.encoder_att, self.encoder_block_att = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleList([])
        for i in range(4):
            self.shared_layer[str(i)] = nn.ModuleList([eval('resnet_network.layer'+str(i+1)+'[:-1]'), 
                                                       eval('resnet_network.layer'+str(i+1)+'[-1]')])
            
            if i == 0:
                self.encoder_att[str(i)] = nn.ModuleList([self._att_layer(ch[0], 
                                                                          ch[0]//self.expansion,
                                                                          ch[0]).to(self.device) for _ in range(self.task_num)])
            else:
                self.encoder_att[str(i)] = nn.ModuleList([self._att_layer(2*ch[i], 
                                                                            ch[i]//self.expansion, 
                                                                            ch[i]).to(self.device) for _ in range(self.task_num)])
                
            if i < 3:
                self.encoder_block_att.append(self._conv_layer(ch[i], ch[i+1]//self.expansion).to(self.device))
                
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def _att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def _conv_layer(self, in_channel, out_channel):
        from .resnet import conv1x1
        downsample = nn.Sequential(conv1x1(in_channel, self.expansion * out_channel, stride=1),
                                   nn.BatchNorm2d(self.expansion * out_channel))
        if self.expansion == 4:
            from .resnet import Bottleneck
            return Bottleneck(in_channel, out_channel, downsample=downsample)
        else:
            from .resnet import BasicBlock
            return BasicBlock(in_channel, out_channel, downsample=downsample)
        
    def forward(self, inputs):
        s_rep = self.shared_conv(inputs)
        ss_rep = {i: [0]*2 for i in range(4)}
        att_rep = [0]*self.task_num
        for i in range(4):
            for j in range(2):
                if i == 0 and j == 0:
                    sh_rep = s_rep
                elif i != 0 and j == 0:
                    sh_rep = ss_rep[i-1][1]
                else:
                    sh_rep = ss_rep[i][0]
                ss_rep[i][j] = self.shared_layer[str(i)][j](sh_rep)
            
            for tn, task in enumerate(self.task_name):
                if self.forward_task is not None and task != self.forward_task:
                    continue
                if i == 0:
                    att_mask = self.encoder_att[str(i)][tn](ss_rep[i][0])
                else:
                    if ss_rep[i][0].size()[-2:] != att_rep[tn].size()[-2:]:
                        att_rep[tn] = self.down_sampling(att_rep[tn])
                    att_mask = self.encoder_att[str(i)][tn](torch.cat([ss_rep[i][0], att_rep[tn]], dim=1))
                att_rep[tn] = att_mask * ss_rep[i][1]
                if i < 3:
                    att_rep[tn] = self.encoder_block_att[i](att_rep[tn])
                if i == 0:
                    att_rep[tn] = self.down_sampling(att_rep[tn])
        if self.forward_task is None:
            return att_rep
        else:
            return att_rep[self.task_name.index(self.forward_task)]