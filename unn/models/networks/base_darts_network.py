from unn.models.networks.base_network import BaseNetwork

class BaseDartsNetwork(BaseNetwork):

    def __init__(self, cfg):
        super(BaseDartsNetwork, self).__init__(cfg)
        self._arch_parameters = []
        for model in self.darts_model():
            self._arch_parameters.extend(model._arch_parameters)
    

    def arch_parameters(self):
        return self._arch_parameters
