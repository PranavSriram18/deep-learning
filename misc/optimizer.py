from typing import List
from collections import defaultdict

class Optimizer:
    def __init__(self, eta: float, gamma: float, params: List):
        self.eta = eta  # learning rate
        self.gamma = gamma
        self.params = params
        self.prev = {param : torch.zeros_like(param) for param in self.params}

    def step(self):
        for param in self.params:
            curr_update = self.gamma * self.prev[param] + self.eta * param.grad()
            param -= curr_update
            self.prev[param] = curr_update

    def save(self, filename: str):
        params_dict = {
            "eta": self.eta, "gamma" : self.gamma, "params": self.params, "prev": self.prev}
        torch.save(params_dict, filename)

    def load(self, filename: str):
        params_dict = torch.load(filename)
        self.eta = params_dict["eta"]
        self.gamma = params_dict["gamma"]
        self.params = params_dict["params"]
        self.prev = params_dict["prev"]


    
