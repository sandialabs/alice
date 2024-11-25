""" Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.

This algorithm implements methods described in the paper, "Curvature in the Looking-Glass:
Optimal Methods to Exploit Curvature of Expectation in the Loss Landscape."
"""
import torch
import alice11_fused_cuda as alice_fused

class Alice(torch.optim.Optimizer):
    """
    This optimization algorithm implements techniques from the paper,
    "Curvature in the Looking-Glass: Optimal Methods to Exploit Curvature of Expectation in the Loss Landscape."

    Args:
        params    : Model parameters,
        lr        : Perturbation length for loss topography and Adam learning rate for maximum step size,
        betas     : Adam betas,
        eps       : Adam epsilon,
        w1        : w1 regularization, w1 || theta ||_1,
        w2        : w2 regularization, w2 0.5 || theta ||_2^2,
        phi       : Fraction of quasi-Newton step. None (default) automatically becomes phi = 1 - betas(0).
        omega     : Look ahead multiplier,
        limit_method : Learning rate limitation method for using lr_min and lr_max, {'fixed', 'sgdm', 'adam' (default)},
        lr_min    : Minimum learning rate using limiter method,
        lr_max    : Maximum learning rate using limiter method. None (default) automatically is replaced by 2.0 * lr.
        scale_bounds_with_lr : True automatically changes group lr_min/max by the same factor as lr.
        hess_comp  : Type of hessian computation, {'zero', 'abs', 'rms'}.
        grad_glass : Include gradient variation in loss extrapolation, {True, False},
        device    : {torch.device('cpu'), torch.device('cuda')}
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        w1=0.,
        w2=0.,
        phi=None,
        omega=1.,
        limit_method='adam',
        lr_min=0.,
        lr_max=None, # None automatically adjusts to 2.0 * lr.
        scale_bounds_with_lr=True, # True -> changing group lr will change lr_min/max by the same factor. 
        hess_comp='abs',
        grad_glass=True,
        quick_steps=3,
        device=torch.device('cuda'),
    ):
        if not lr > 0:
            raise ValueError(f"The learning rate {lr} must be positive.")
        if not 0 < betas[0] < 1:
            raise ValueError(f"The average coefficient beta1 {beta[0]} must be between 0 and 1.")
        if not 0 < betas[1] < 1:
            raise ValueError(f"The average coefficient beta2 {beta[1]} must be between 0 and 1.")
        if not 0 < eps:
            raise ValueError(f"The denominator coefficient eps {eps} must be greater than 0.")
        if not 0 <= w1:
            raise ValueError(f"The w1 regularization coefficient {w1} must be nonnegative.")
        if not 0 <= w2:
            raise ValueError(f"The w2 regularization coefficient {w2} must be nonnegative.")
        if phi == None:
            phi = 1 - betas[0]
        elif not 0 < phi < 2:
            raise ValueError(f"The quasi-Newton step fraction {phi} must be positive and less than 2.")
        if not 0 < omega:
            raise ValueError(f"The look-ahead coefficient {omega} must be positive.")
        if limit_method == 'fixed':
            limiter = 0
        elif limit_method == 'sgdm':
            limiter = 1
        elif limit_method == 'adam':
            limiter = 2
        else:
            raise ValueError(f"Unknown limiter method {limit_method}")
        if not 0 <= lr_min:
            raise ValueError(f"The minimum learning rate {lr_min} must be nonnegative.")
        if lr_max == None:
            lr_max = 2.0*lr
        if not lr_min <= lr_max:
            raise ValueError(f"The maximum learning rate {lr_max} must be greater than or equal to the minimum {lr_min}.")
        if not (quick_steps >= 0 and quick_steps == int(quick_steps)):
            raise ValueError(f"The number of quick steps {quick_steps} must be a nonnegative integer.")
        if hess_comp == 'zero':
            hc = 0
        elif hess_comp == 'abs':
            hc = 1
        elif hess_comp == 'rms':
            hc = 2
        else:
            raise ValueError(f"Unknown hessian computation type {hess_comp}")
        if grad_glass == False and hc == 0:
            raise ValueError(f"Either the grad_glass {grad_glass} or hessian computation {hess_comp} must be nonzero.")
        # Set hyperparamenters that can be customized for each parameter group.
        defaults = dict(lr=torch.tensor(lr, device=device), lr_min=torch.tensor(lr_min, device=device), lr_max=torch.tensor(lr_max, device=device))
        super().__init__(params, defaults)

        # State matrix indices:
        self.mMu  = 0 # parameter center.
        self.mNu  = 1 # look-ahead location.
        self.mZ   = 2 # temporary
        self.mG   = 3 # running average gradient
        self.mR   = 4 # running average gradient variance density
        self.mH   = 5 # running average hessian (absolute value or squared)
        self.mV   = 6 # running variance
        self.mLen = 7

        # Shared float indices:
        self.fSig  = 0
        self.fBet1 = 1
        self.fBet2 = 2
        self.fBet3 = 3
        self.fEps  = 4
        self.fW1   = 5
        self.fW2   = 6
        self.fPhi  = 7
        self.fOmg  = 8
        self.fTau1 = 9
        self.fTau2 = 10
        self.fLen  = 11
 
        # Shared int indices:
        self.iQ   = 0
        self.iH   = 1
        self.iR   = 2
        self.iL   = 3
        self.iLen = 4

        self.device = device
        self.F = torch.zeros((self.fLen,), device=self.device, dtype=torch.float, requires_grad=False)
        self.I = torch.zeros((self.iLen,), device=self.device, dtype=torch.int32, requires_grad=False)

        self.scale_bounds = scale_bounds_with_lr
        self.quick_steps = quick_steps
        self.step_period = quick_steps + 1
        self.beta1 = torch.tensor(betas[0], device=self.device, requires_grad=False)
        self.beta2 = torch.tensor(betas[1], device=self.device, requires_grad=False)
        self.beta3 = torch.tensor(1. - self.step_period * (1. - betas[1]), device=self.device, requires_grad=False)
        self.F[self.fEps] = eps
        self.F[self.fW1] = w1
        self.F[self.fW2] = w2
        self.F[self.fPhi] = phi
        self.F[self.fOmg] = omega

        self.I[self.iH] = hc
        self.I[self.iR] = grad_glass
        self.I[self.iL] = limiter

        # Initialize group and parameter variables
        self.reset()
        print('num_par = {}'.format(self.num_par))

    def state_str(self):
        ret_str = "Alice 1.1: Basic optimization with glass curvature. Allow fixed, SGDM, and Adam interpretations of lr_min and lr_max."
        ret_str += f" quick_steps={self.quick_steps}"
        ret_str += f" beta1={self.beta1}"
        ret_str += f" beta2={self.beta2}"
        ret_str += f" beta3={self.beta3}"
        ret_str += f" eps={self.F[self.fEps]}"
        ret_str += f" w1={self.F[self.fW1]}"
        ret_str += f" w2={self.F[self.fW2]}"
        ret_str += f" phi={self.F[self.fPhi]}"
        ret_str += f" omega={self.F[self.fOmg]}"
        ret_str += f" limit_method={self.I[self.iL]}"
        ret_str += f" lr_min={self.F[self.fTau1]}"
        ret_str += f" lr_max={self.F[self.fTau2]}"
        for i, grp in enumerate(self.param_groups):
            ret_str += f" group_{i}: ["
            ret_str += f" lr={grp['lr']}"
            if self.scale_bounds:
                ret_str += f" scale_min={grp['scale_min']}"
                ret_str += f" scale_max={grp['scale_max']}"
            else:
                ret_str += f" lr_min={grp['lr_min']}"
                ret_str += f" lr_max={grp['lr_max']}"
            ret_str += f" ]"
        ret_str += f" hess_comp={self.I[self.iH]}"
        ret_str += f" grad_glass={self.I[self.iR]}"
        return ret_str

    def reset(self):
        self.num_par = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.num_batch = torch.tensor(0, dtype=torch.int32, device=self.device)
        for grp in self.param_groups:
            if self.scale_bounds:
                grp['scale_min'] = grp['lr_min'] / grp['lr']
                grp['scale_max'] = grp['lr_max'] / grp['lr']
            for p in grp['params']:
                if p.requires_grad:
                    state = self.state[p]
                    self.num_par += p.numel()
                    if not 'M' in state:
                        # Create state matrix.
                        state['M'] = torch.zeros((p.numel(), self.mLen), dtype=p.dtype, device=p.device, requires_grad=False)
                    # Set mean and look-ahead to current parameter values
                    state['M'][:, self.mMu] = p.data.flatten()
                    state['M'][:, self.mNu] = p.data.flatten()

    def center(self):
        with torch.no_grad():
            jc = 0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        p.data = state['M'][:, self.mMu].detach().clone().view(p.size())

    def _init_pert(self):
        with torch.no_grad():
            for grp in self.param_groups:
                self.F[self.fSig] = grp['lr']
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        # Alg. 2, Step 1. This is the first step in drawing Rademacher samples. 
                        # See fused kernel for more details.
                        state['M'][:, self.mZ] = torch.rand(p.numel(), dtype=p.dtype, device=p.device)
                        alice_fused.init_pert(p.data, state['M'], self.F)

    def _pert_update(self, sample_index):
        with torch.no_grad():
            if sample_index > 2:
                raise RuntimeError(f"This version only supports 2 samples.")
            self.I[self.iQ] = sample_index
            for grp in self.param_groups:
                self.F[self.fSig] = grp['lr']
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        # Alg. 2, Steps 1, 3, and 7. See fused kernel for more details.
                        if p.grad == None:
                            alice_fused.pert_update(p.data, state['M'], torch.zeros_like(p.data), self.F, self.I)
                        else:
                            alice_fused.pert_update(p.data, state['M'], p.grad, self.F, self.I)

    def _state_update(self):
        with torch.no_grad():
            for grp in self.param_groups:
                self.F[self.fSig] = grp['lr']
                if self.scale_bounds:
                    self.F[self.fTau1] = grp['lr'] * grp['scale_min']
                    self.F[self.fTau2] = grp['lr'] * grp['scale_max']
                else:
                    self.F[self.fTau1] = grp['lr_min']
                    self.F[self.fTau2] = grp['lr_max']
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        # Alg 3. See fused kernel for more details.
                        alice_fused.state_update(p.data, state['M'], self.F, self.I)

    def _quick_step(self, model_func, loss_func):
        # Set parameter to nu.
        with torch.no_grad():
            jc = 0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        # Set parameters to unperturbed nu.
                        p.data = state['M'][:, self.mNu].detach().clone().view(p.size())
        # Evaluate the loss and gradient
        self.zero_grad()
        outputs = model_func()
        loss = loss_func(outputs)
        loss.backward()
        # Update running average gradient and second moment.
        with torch.no_grad():
            for grp in self.param_groups:
                self.F[self.fSig] = grp['lr']
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        # Implements Adam update of running gradient and second moment. See fused kernel for details.
                        if p.grad == None:
                            alice_fused.quick_update(p.data, state['M'], torch.zeros_like(p.data), self.F, self.I)
                        else:
                            alice_fused.quick_update(p.data, state['M'], p.grad, self.F, self.I)
        return loss, outputs

    def _sample_step(self, model_func, loss_func):
        self._init_pert()
        # Save rng state before first evaluation.
        if self.device == torch.device('cuda'):
            model_rng_state = torch.cuda.get_rng_state()
        else:
            model_rng_state = torch.random.get_rng_state()
        for s in range(3):
            # Set rng state to replicate with each perturbation.
            if self.device == torch.device('cuda'):
                torch.cuda.set_rng_state(model_rng_state)
            else:
                torch.random.set_rng_state(model_rng_state)
            # Alg 2, Steps 2, 4, and 8.
            self.zero_grad()
            outputs = model_func()
            loss = loss_func(outputs)
            loss.backward()
            self._pert_update(s)
        return loss, outputs

    def step(self, closure):
        """ closure = (model_func, loss_func)
            model_func() : This function evaluates the model and returns the outputs.
            loss_func(outputs) : This function accepts model outputs and evaluates the loss criterion.

            Neither model_func() nor loss_func(outputs) need to zero gradients or call backpropagation. """
        self._cuda_graph_capture_health_check()
        (model_func, loss_func) = closure

        # Increment number of batches and update betas accordingly.
        self.num_batch += 1
        beta_max = (self.num_batch - 1.)/self.num_batch
        self.F[self.fBet1] = torch.minimum(self.beta1, beta_max)
        self.F[self.fBet2] = torch.minimum(self.beta2, beta_max)
        self.F[self.fBet3] = torch.minimum(self.beta3, beta_max)

        if self.num_batch <= 1./(1. - self.beta1) or self.num_batch % self.step_period == 0:
            loss, outputs = self._sample_step(model_func, loss_func)
        else:
            loss, outputs = self._quick_step(model_func, loss_func)

        # Build the running average to the effective sample size before stepping.
        if self.num_batch > 1./(1. - self.beta1):
            self._state_update()

        # Always leave the model centered at mu.
        self.center()
        return loss, outputs

    def export_mu(self):
        with torch.no_grad():
            mu = []
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        mu.append(state['M'][:, self.mMu].detach().clone().view(p.size()))
        return mu

    def import_mu(self, mu):
        with torch.no_grad():
            j = 0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        p.data = mu[j].detach().clone()
                        j += 1

    def get_attributes(self):
        with torch.no_grad():
            rho = []
            h = []
            for grp in self.param_groups:
                rho_ = torch.tensor(0., dtype=torch.float, device=self.device)
                h_ = torch.tensor(0., dtype=torch.float, device=self.device)
                np_ = torch.tensor(0., dtype=torch.float, device=self.device)
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        rho_ += state['M'][:, self.mR].sum()
                        h_ += state['M'][:, self.mH].sum()
                        np_ += p.numel()
                rho.append(rho_ / np_)
                h.append(h_ / np_)
        return rho, h

