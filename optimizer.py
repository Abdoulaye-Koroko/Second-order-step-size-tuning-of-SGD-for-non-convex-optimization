import torch
from torch.optim.optimizer import Optimizer

class step_tuned_SGD(Optimizer):
    
    """Implement Second-order step-size tuning of SGD algorithm https://arxiv.org/pdf/2103.03570v1.pdf
    
    Inputs:
        - params: model parameters to optimize
        - alpha (float): inital learning rate
        - mu (float): large step-size to use in locally concave regions.
        - beta (float): exponentionally weighting parameter
        - m and M (floats): interval that should contains the computed learning rate.
        - delta (float): parameter used to enforce the learning rate to decrease.
        
        
        
    Implemented by Abdoulaye KOROKO.
        
    
    """
    
    
    def __init__(self,params,alpha,mu=2.0,beta=0.9,m=0.5,M=2.0,delta=0.501):
        
         if alpha < 0.0:
            raise ValueError("Invalid initial learning rate. Must be greater than 0") 
        if mu<0:
            raise ValueError("Invalid mu value. Must be greater than 0")
        if beta<0:
            raise ValueError("Invalid beta value. Must be greater than 0.")
        if m<0 or M<0:
            raise ValueError("Invalid values for m and M. Must be greater than 0")
        if delta<0:
            raise ValueError("Invalid value for delta. Must be greater than 0")
        
        defaults = dict(alpha=alpha,mu=mu,beta=beta,m=m,M=M,delta=delta)
        super(step_tuned_SGD,self).__init__(params,defaults)
        
        for group in self.param_groups:
            group.update({"_lambda":1,"G":0})
            
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                
                
    def copy_flat_params(self):
        "Copy all the model parameters and flattens them into a vector"
        
        parameters=[]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                parameters.append(deepcopy(p.data).view(-1))
        return torch.cat(parameters,0)
    
    def gather_flat_grad(self):
        "Copy the gradients of all the model parameters and flattens them into a vector"
        grads = []
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grads.append(p.grad.data.view(-1))
        return torch.cat(grads,0)
    
    
    def udpdate_params(self,alpha,_lambda,delta,k,directions):
        "Computes the tuned-learning rate and updates the model parameters given the input values"
        lr = (alpha*_lambda)/((k+1)**delta)
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                numel=p.numel()
                p.data.add_(other=directions[offset:offset + numel].view_as(p.data),alpha=-lr)
                offset += numel
        return
    
    def get_state_step(self):
        "Gets the states of the model paraemters"
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                state =self.state[p]
                return state["step"]
    def update_step(self):
        "Update step"
        
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                state =self.state[p]
                state["step"]+=1
        return
        
    
    
    def step(self,closure):
        
         """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        
        loss = None
        
        if closure is None:
            raise ValueError("This optimizer requires closure function")
        
        loss = closure()
        
        for group in self.param_groups:
            alpha = group["alpha"]
            mu = group["mu"]
            beta = group["beta"]
            m = group["m"]
            M = group["M"]
            delta = group["delta"]
            _lambda = group["_lambda"]
            G = group["G"]
            
            k = self.get_state_step()
            grads_k = self.gather_flat_grad()
            params_k = self.copy_flat_params()
            self.udpdate_params(alpha,_lambda,delta,k,grads_k)
            new_loss = closure()
            new_grads = self.gather_flat_grad()
            new_params = self.copy_flat_params()
            self.udpdate_params(alpha,_lambda,delta,k,new_grads)
            delta_params = new_params-params_k
            delta_grads = new_grads-grads_k
            G = beta*G+(1-beta)*delta_grads
            G_hat = G/(1-beta**(k+1))
            
            test = G_hat.dot(delta_params)
            if test>0:
                _lambda = delta_params.dot(delta_params)/test
            else:
                _lambda = mu
                
            
            _lambda = min(max(_lambda,m),M)
            
            group["_lambda"] = _lambda
            group["G"] = G
            
            self.update_step()
            
        return new_loss    
