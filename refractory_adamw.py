import torch
from torch.optim import Optimizer
import math

class RefractoryAdamW(Optimizer):
    """
    AdamW with parameter-level refractory period.
    
    Each parameter has a 'refractory state' that decays naturally over time.
    When a parameter updates significantly, its refractory state increases,
    reducing subsequent updates. The state decays exponentially back to zero.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, refractory_decay=0.99, 
                 refractory_scale=1.0, min_refractory=0.1):
        """
        Args:
            refractory_decay: How quickly refractory state decays (0.9-0.999)
            refractory_scale: How much large updates increase refractory state
            min_refractory: Minimum refractory multiplier (prevents complete freezing)
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        refractory_decay=refractory_decay,
                        refractory_scale=refractory_scale,
                        min_refractory=min_refractory)
        super(RefractoryAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            refractory_decay = group['refractory_decay']
            refractory_scale = group['refractory_scale']
            min_refractory = group['min_refractory']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['refractory'] = torch.ones_like(p)  # 1.0 = normal

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                refractory = state['refractory']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Standard Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                update = exp_avg / denom
                
                # Apply weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # --- Refractory mechanism ---
                # Decay refractory state (move back toward 1.0)
                refractory.mul_(refractory_decay).add_(1 - refractory_decay)
                
                # Increase refractory state where update is large
                update_norm = torch.abs(update)
                # Normalize by parameter magnitude to make it scale-invariant
                param_norm = torch.abs(p) + eps
                relative_update = update_norm / param_norm
                
                # Increase refractory more for larger updates
                refractory.add_(relative_update * refractory_scale)
                
                # Clamp to reasonable range
                refractory.clamp_(min=min_refractory, max=1.0)
                
                # Apply update with refractory damping
                p.add_(-step_size * update / refractory)
                
                # Alternative: also track update history for better decay
                if 'last_update' not in state:
                    state['last_update'] = torch.zeros_like(p)
                state['last_update'] = update.clone()

        return loss