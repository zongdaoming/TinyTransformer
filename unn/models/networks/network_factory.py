from unn.models.networks.base_network import BaseNetwork
from unn.models.networks.darts.backbone_darts_network import BackboneDartsNetwork

class NetworkFactory:

    @classmethod
    def create(cls, name, cfg):
        if name == 'base':
            network = BaseNetwork(cfg)
        elif name == 'backbone_darts':
            network = BackboneDartsNetwork(cfg)
        else:
            raise ValueError('Unrecognized Network: ' + name)
        return network

