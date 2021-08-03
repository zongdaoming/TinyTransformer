from unn.models.networks.base_darts_network import BaseDartsNetwork

class BackboneDartsNetwork(BaseDartsNetwork):

    def __init__(self, cfg):
        super(BackboneDartsNetwork, self).__init__(cfg)

    def darts_model(self):
        return [self.backbone]
