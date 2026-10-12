"""Optimizer IO for D's AdamW/SGD control, following the 10/08 branch split."""
import torch


class CellAdamWODESGD:
    def __init__(self, model, config):
        cell=list(model.ml_model.parameters());ode=list(model.ode_model.parameters())
        a,b=set(map(id,cell)),set(map(id,ode))
        if a & b or a | b != set(map(id,model.parameters())):
            raise ValueError('Optimizer branches must be disjoint and exhaustive')
        self.cell=torch.optim.AdamW(cell,lr=config['lr'],weight_decay=config['weight_decay'])
        self.ode=torch.optim.SGD(ode,lr=config['lr'],weight_decay=config['weight_decay'],momentum=0.,nesterov=False)
        self.param_groups=self.cell.param_groups+self.ode.param_groups

    def zero_grad(self, set_to_none=True):
        self.cell.zero_grad(set_to_none=set_to_none)
        self.ode.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.cell.step();self.ode.step()

    def state_dict(self):
        return dict(schema='cell_adamw_ode_sgd_v1',cell=self.cell.state_dict(),ode=self.ode.state_dict())

    def load_state_dict(self, state):
        if state.get('schema')!='cell_adamw_ode_sgd_v1':
            raise ValueError('Expected dual optimizer checkpoint; cannot resume shared AdamW')
        self.cell.load_state_dict(state['cell']);self.ode.load_state_dict(state['ode'])
        self.param_groups=self.cell.param_groups+self.ode.param_groups
