import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize

class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self, model, task_num=10, device='cuda', rep=None, rep_tasks=None):
        super(AbsWeighting, self).__init__()
        self.model = model
        self.task_num = 10
        self.device = device
        self.rep = rep
        self.rep_tasks = rep_tasks
        self.rep_grad = False
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.model.parameters():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(10, self.grad_dim).to(self.device)
            for tn in range(10):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=10 else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.model.parameters(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(10, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=10 else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(10, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(10, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=10 else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass

class GradVac(AbsWeighting):
    r"""Gradient Vaccine (GradVac).
    
    This method is proposed in `Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models (ICLR 2021 Spotlight) <https://openreview.net/forum?id=F1vEjWK-lH_>`_ \
    and implemented by us.

    Args:
        beta (float, default=0.5): The exponential moving average (EMA) decay parameter.

    .. warning::
            GradVac is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(GradVac, self).__init__()
        
    def init_param(self):
        self.rho_T = torch.zeros(10, 10).to(self.device)
        
    def backward(self, losses, **kwargs):
        beta = kwargs['beta']

        if self.rep_grad:
            raise ValueError('No support method GradVac with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(10):
            task_index = list(range(10))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (pc_grads[tn_i].norm()*grads[tn_j].norm())
                if rho_ij < self.rho_T[tn_i, tn_j]:
                    w = pc_grads[tn_i].norm()*(self.rho_T[tn_i, tn_j]*(1-rho_ij**2).sqrt()-rho_ij*(1-self.rho_T[tn_i, tn_j]**2).sqrt())/(grads[tn_j].norm()*(1-self.rho_T[tn_i, tn_j]**2).sqrt())
                    pc_grads[tn_i] += grads[tn_j]*w
                    batch_weight[tn_j] += w.item()
                    self.rho_T[tn_i, tn_j] = (1-beta)*self.rho_T[tn_i, tn_j] + beta*rho_ij
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight

class CAGrad(AbsWeighting):
    r"""Conflict-Averse Gradient descent (CAGrad).
    
    This method is proposed in `Conflict-Averse Gradient Descent for Multi-task learning (NeurIPS 2021) <https://openreview.net/forum?id=_61Qh8tULj_>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/Cranial-XIX/CAGrad>`_. 

    Args:
        calpha (float, default=0.5): A hyperparameter that controls the convergence rate.
        rescale ({0, 1, 2}, default=1): The type of the gradient rescaling.

    .. warning::
            CAGrad is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(CAGrad, self).__init__()
        
    def backward(self, losses, **kwargs):
        calpha, rescale = kwargs['calpha'], kwargs['rescale']

        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')
        
        GG = torch.matmul(grads, grads.t()).cpu() # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

        x_start = np.ones(10) / 10
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (calpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,-1).dot(A).dot(b.reshape(-1,1))+c*np.sqrt(x.reshape(1,-1).dot(A).dot(x.reshape(-1,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(self.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = grads.mean(0) + lmbda * gw
        if rescale == 0:
            new_grads = g
        elif rescale == 1:
            new_grads = g / (1+calpha**2)
        elif rescale == 2:
            new_grads = g / (1 + calpha)
        else:
            raise ValueError('No support rescale type {}'.format(rescale))
        self._reset_grad(new_grads)
        return w_cpu