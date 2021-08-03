from unn.models.archs.base_arch import BaseArch
from unn.models.archs.darts_arch import DartsArch

class ArchFactory:

    @classmethod
    def create(cls, name, cfg):
        if name == 'base':
            arch = BaseArch(cfg)
        elif name == 'darts':
            arch = DartsArch(cfg)
        else:
            raise ValueError('Unrecognized Arch: ' + name)
        return arch
