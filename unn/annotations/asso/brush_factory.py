from .human_parts_brush import HumanPartsBrush

class BrushFactory:

    @classmethod
    def create(cls, cfg):

        method = cfg['method']
        if 'kwargs' not in cfg:
            cfg['kwargs'] = {}
        kwargs = cfg['kwargs']
        if method == 'human parts':
            brush = HumanPartsBrush(**kwargs)
        else:
            raise ValueError('Unrecognized Brush Method For Association: ' + method)
        return brush

