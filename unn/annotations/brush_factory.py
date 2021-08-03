from .asso.brush_factory import BrushFactory as AssoBrushFactory
from .bbox.brush_factory import BrushFactory as BboxBrushFactory
from .keyp.brush_factory import BrushFactory as KeypBrushFactory

class BrushFactory:

    @classmethod
    def create(cls, name, cfg):

        if name == 'asso':
            brush = AssoBrushFactory.create(cfg)
        elif name == 'bbox':
            brush = BboxBrushFactory.create(cfg)
        elif name == 'keyp':
            brush = KeypBrushFactory.create(cfg)
        else:
            raise ValueError('Unrecognized Annotation for visualization: ' + name)
        return brush

