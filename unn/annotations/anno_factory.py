import logging
from .asso.asso_annotation import AssoAnnotation
from .bbox.bbox_annotation import BboxAnnotation
from .image.image_annotation import ImageAnnotation
from .keyp.keyp_annotation import KeypAnnotation
import pdb
logger = logging.getLogger('global')

class AnnoFactory:

    @classmethod
    def create(cls, name, cfg):

        if name == 'image':
            anno = ImageAnnotation(cfg)
        elif name == 'asso':
            anno = AssoAnnotation(cfg)
        elif name == 'bbox':
            anno = BboxAnnotation(cfg)
        elif name == 'keyp':
            anno = KeypAnnotation(cfg)
        else:
            raise ValueError('Unrecognized Annotation: ' + name)
        return anno
