from unn.models.networks.network_factory import NetworkFactory
from unn.utils.dist_helper import get_world_size

class BaseArch:

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = NetworkFactory.create(self.cfg['network']['name'], self.cfg['network']['cfg'])

    def forward(self, input):
        return self.model(input)

    def new(self):
        model_new = NetworkFactory.create(self.cfg['network']['name'], self.cfg['network']['cfg']).cuda()
        return model_new
