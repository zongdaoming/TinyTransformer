from .general_brush import GeneralBrush

class BrushFactory:

    @classmethod
    def create(cls, cfg):

        method = cfg['method']
        if 'kwargs' not in cfg:
            cfg['kwargs'] = {}
        kwargs = cfg['kwargs']
        if method == 'general':
            brush = GeneralBrush(**kwargs)
        else:
            raise ValueError('Unrecognized Brush Method For Bounding Box: ' + method)
        return brush

