import os
import yaml
import re
from collections import OrderedDict
from .. import version

model_dir = '/mnt/lustre/share/DSK/model_zoo/pytorch/imagenet'
model_zoo = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
    'resnext_101_32x4d': 'resnext_101_32x4d.pth',
    'resnext_101_32x8d': 'resnext_101_32x8d.pth',
    'resnext_101_64x4d': 'resnext_101_64x4d.pth',
    'resnext_101_64x8d': 'resnext_101_64x8d.pth',
    'resnext_152_32x4d': 'resnext_152_32x4d.pth',
    'resnext_152_32x8d': 'resnext_152_32x8d.pth',
    'resnext_152_64x4d': 'resnext_152_64x4d.pth',
    'resnext_152_64x8d': 'resnext_152_64x8d.pth',
    'senet154': 'senet154-c7b49a05.pth',
    'se_resnet50': 'se_resnet50-ce0d4300.pth',
    'se_resnet101': 'se_resnet101-7e38fcc6.pth',
    'se_resnet152': 'se_resnet152.pth',
    'se_resnext50_32x4d': 'se_resnext50_32x4d.pth',
    'se_resnext101_32x4d': 'se_resnext101_32x4d.pth',
    'se_resnext101_64x4d': 'se_resnext101_64x4d.pth',
    'nasnetAlarge6_3072': 'nasnet6@3072.pth.tar',
    'bqnnv1_large': 'BlockQNN/BlockQNNv1-depthwise_v1.1-ImageNet-large-91.53M-81.01-37cb2732ad.pth',
    'mobilenetv2': 'mobilenet_v2/mobilenet_v2_1.0.pth',
    'shufflenetv2': 'shufflenet_v2/shufflenetv2_x1.0_top168.pth.tar'
}


class Args(object):
    def __init__(self,
                 dataset='COCO',
                 num_classes=0,
                 backbone='resnet50',
                 feature='FPN',
                 first_stage='RPN',
                 second_stage=None,
                 multi_task_stage=None,
                 IterMultiplier=1,
                 OHEM=False,
                 DCNv2=False,
                 L1=False,
                 FP16=False,
                 SoftNMS=False,
                 MSTest=False,
                 SyncBN=False):
        self.dataset = dataset
        self.num_classes = num_classes
        self.backbone = backbone
        self.feature = feature
        self.first_stage = first_stage
        self.second_stage = second_stage
        self.multi_task_stage = multi_task_stage
        self.IterMultiplier = IterMultiplier
        self.OHEM = OHEM
        self.DCNv2 = DCNv2
        self.L1 = L1
        self.FP16 = FP16
        self.SoftNMS = SoftNMS
        self.MSTest = MSTest
        self.SyncBN = SyncBN


def ordered_yaml_load(yaml_path, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    with open(yaml_path) as stream:
        return yaml.load(stream, OrderedLoader)


def ordered_yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def get_path(yml_name):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(project_dir, 'configs/components', yml_name)


dataset_yml = get_path('dataset.yml')
trainer_yml = get_path('trainer.yml')
backbone_yml = get_path('backbone.yml')
feature_yml = get_path('feature.yml')
first_stage_yml = get_path('first_stage.yml')
second_stage_yml = get_path('second_stage.yml')
multi_task_yml = get_path('multi_task_stage.yml')


def load_yml(yml_path):
    return ordered_yaml_load(yml_path)


def get_dataset(dataset, multi_task_stage, num_classes=0):
    """
    Args:
        dataset: dataset type
        num_classes: number of classes
    Return:
        dataset: config of dataset
    """
    datas = load_yml(dataset_yml)
    if multi_task_stage == 'Keypoint':
        dataset = datas['COCO-Keypoint']
    else:
        dataset = datas[dataset]
    if dataset['type'] == 'custom':
        assert num_classes != 0, 'num_classes is required for custom dataset'
        dataset['num_classes'] = num_classes
    return dataset


def get_trainer(dataset, iter_multiplier, multi_task_stage):
    """
    Args:
        dataset: config of dataset
    Return:
        trainer: config of trainer
    """
    trainers = load_yml(trainer_yml)
    if dataset['type'] == 'coco':
        if multi_task_stage == 'Grid':
            trainer = trainers['COCO-Grid']
        elif multi_task_stage == 'Keypoint':
            trainer = trainers['COCO-Keypoint']
        else:
            trainer = trainers['COCO-Keypoint']
    else:
        trainer = trainers['VOC']

    return trainer


def get_saver(backbone):
    """backbone: callable name to build backbone"""
    saver = {'save_dir': 'checkpoints', 'results_dir': 'results_dir'}
    if backbone not in model_zoo:
        raise ValueError(f'{backbone} is not supported yet')
    pretrain_path = os.path.join(model_dir, model_zoo[backbone])
    if not os.path.exists(pretrain_path):
        print('****************warning*******************' f'There is no pre-trained model for {backbone}')
    else:
        saver['pretrain_model'] = pretrain_path
    return saver


def get_backbone(backbone, feature, first_stage):
    """ valid (backbone, feature) combinations are (resnet, FPN/C4/C5)
    and (renext/senet/nasnet/bqnn/mobilenet/shffulent, FPN)
    """
    backbones = load_yml(backbone_yml)
    if feature != 'FPN':
        assert backbone.startswith('resnet'), f"{backbone} do not support {feature} yet"
    for name, cfg in backbones.items():
        bname, fname = name.split('-')
        if fname == feature and backbone.lower().startswith(bname.lower()):
            # print(f'{backbone}-{feature} matches {name}')
            backbone = cfg['type'].rsplit('.', 1)[0] + '.' + backbone
            cfg['type'] = backbone
            return cfg
        if fname == first_stage and backbone.lower().startswith(bname.lower()):
            # print(f'{backbone}-{feature} matches {name}')
            backbone = cfg['type'].rsplit('.', 1)[0] + '.' + backbone
            cfg['type'] = backbone
            return cfg
    raise NotImplementedError(f'{backbone}-{feature} is not supported yet')


def get_feature(feature, first_stage):
    """FPN, C4, C5"""
    cfg = load_yml(feature_yml)
    key = f'{feature}-{first_stage}'
    return cfg.get(key, None)


def get_first_stage(first_stage, feature, num_classes, dataset):
    fs = load_yml(first_stage_yml)[first_stage]
    # setup classification
    if first_stage == 'RetinaNet':
        fs['kwargs']['num_classes'] = num_classes

    # update feature
    if feature == 'FPN':
        fs['prev'] = 'neck'
    else:
        fs['prev'] = 'backbone'

    # update anchor setting
    if feature != 'FPN':
        if dataset['type'] == 'pascal_voc':
            fs['kwargs']['cfg']['anchor_scales'] = [8, 16, 32]

        else:
            fs['kwargs']['cfg']['anchor_scales'] = [2, 4, 8, 16, 32]
        fs['kwargs']['cfg']['train'].pop('across_levels')
        fs['kwargs']['cfg']['test'].pop('across_levels')
        fs['kwargs']['cfg']['train']['pre_nms_top_n'] = 12000
        fs['kwargs']['cfg']['train']['post_nms_top_n'] = 2000
        fs['kwargs']['cfg']['test']['pre_nms_top_n'] = 6000
        fs['kwargs']['cfg']['test']['post_nms_top_n'] = 300
    return fs


def get_second_stage(second_stage, num_classes, feature, backbone=None):
    if not second_stage:
        return None
    ss = load_yml(second_stage_yml)[second_stage]
    if second_stage == 'Res5':
        ss['kwargs']['backbone'] = backbone
    ss['kwargs']['num_classes'] = num_classes
    if feature == 'FPN':
        ss['prev'] = 'neck'
        ss['kwargs']['cfg']['fpn'] = {'fpn_levels': [0, 1, 2, 3], 'base_scale': 56}
    elif feature == 'C4' or feature == 'C5':
        ss['prev'] = 'backbone'
        ss['kwargs']['cfg'].pop('fpn', None)
    else:
        raise NotImplementedError(f'feature:{feature} is not supported yet')
    return ss


def get_multi_task(multi_task, num_classes, feature):
    if not multi_task:
        return None
    mt = load_yml(multi_task_yml)[multi_task]
    if multi_task == 'Mask':
        mt['kwargs']['num_classes'] = num_classes
    if feature == 'FPN':
        mt['prev'] = 'neck'
        mt['kwargs']['cfg']['fpn'] = {'fpn_levels': [0, 1, 2, 3], 'base_scale': 56}
    elif feature == 'C4' or feature == 'C5':
        mt['prev'] = 'backbone'
        mt['kwargs']['cfg'].pop('fpn', None)
    else:
        raise NotImplementedError(f'feature:{feature} is not supported yet')

    return mt


def get_fp16(fp16):
    if fp16:
        return {'scale_factor': 1024}
    else:
        return None


def dcnv2(backbone, second_stage):
    """Add Deformable Conv to backbone and second stage head """

    backbone_name = backbone['type'].rsplit('.', 1)[-1]
    assert backbone_name.startswith('res'), f'DCNv2 only support ResNet && ResNeXt, but {backbone_name} found'
    pattern = re.compile(r'\D*(?P<depth>\d+)')
    res = pattern.match(backbone_name)
    assert res, 'there is no depth information in backbone name'
    depth = int(res.groupdict()['depth'])
    if depth > 50:
        layer_deform = [False, False, 'all', 8, 'all']
    else:
        layer_deform = [False, False, 'all', 'all', 'all']
    backbone['kwargs']['layer_deform'] = layer_deform

    if second_stage and second_stage['type'].endswith('Res5'):
        second_stage['kwargs']['deformable'] = True
    return True


def ohem(second_stage):
    """Add OHEM to second stage head"""
    if second_stage is None:
        print('there is not second stage for this detector, OHEM can not be used')
        return False
    ohem = OrderedDict({'batch_size': 512})
    sampler = OrderedDict({'type': 'keep_all'})
    second_stage['kwargs']['cfg']['train']['ohem'] = ohem
    second_stage['kwargs']['cfg']['train']['sampler'] = sampler
    return True


def softnms(second_stage):
    nms = OrderedDict({
        'type': 'soft',
        'nms_iou_thresh': 0.5,
        'softnms_sigma': 0.5,
        'softnms_bbox_score_thresh': 0.0001,
        'softnms_method': 'linear'
    })
    if second_stage is not None:
        second_stage['kwargs']['cfg']['test']['nms'] = nms
        return True
    else:
        print("there is no second stage for this detector,"
              "if you actually need to use softnms in single stage detector, pls configure it manually")
        return False


def second_stage_grid_flag(multi_task, second_stage):
    if multi_task == 'Grid':
        assert second_stage is not None
        second_stage['kwargs']['cfg']['grid'] = True
    return second_stage


def multi_scale_test(dataset, second_stage, multi_task):
    assert not multi_task or multi_task == 'Grid', 'Multi-scale test do not support mask & keypoint yet'
    # modify dataset
    dataset['test']['batch_size'] = 1
    dataset['test']['scales'] = [-1]
    # ms_test config
    ms_test = OrderedDict({
        'bbox_aug': {
            'scales': [i * 100 for i in range(4, 13)],
            'max_size': 2000,
            'hflip': True,
            'alignment': dataset['alignment']
        },
        'nms': {
            'type': 'naive',
            'nms_iou_thresh': 0.5
        },
        'bbox_score_thresh': 0.05,
        'top_n': 100
    })
    # modify second stage post-process
    if second_stage:
        second_stage['kwargs']['cfg']['test']['nms'] = {'type': 'naive', 'nms_iou_thresh': -1}
        second_stage['kwargs']['cfg']['test']['top_n'] = -1
        second_stage['kwargs']['cfg']['test']['bbox_score_thresh'] = 0.05
    return ms_test


def sync_bn(backbone, second_stage, multi_task_stage):
    bn_cfg = OrderedDict({'sync': {'bn_group_size': 8}})
    if multi_task_stage and multi_task_stage.find('grid') >= 0:
        multi_task_stage['kwargs']['bn'] = bn_cfg
        bn_cfg = {'freeze': True}
    backbone['kwargs']['bn'] = bn_cfg
    if second_stage['type'].endswith('Res5'):
        second_stage['kwargs']['bn'] = bn_cfg
    return True


def update_dataset_output(dataset, multi_task):
    if multi_task == 'Grid':
        dataset['has_grid'] = True
    elif multi_task == 'Mask':
        dataset['has_mask'] = True
        if 'evaluator' in dataset:
            dataset['evaluator']['kwargs']['iou_types'].append('segm')
    elif multi_task == 'Keypoint':
        # assume we are training COCO keypoints
        dataset['has_keypoint'] = True
        dataset['num_classes'] = 2
        dataset['class_names'] == ['__background__', 'person']
        dataset['train']['scales'] = [640, 672, 704, 736, 768, 800]
        dataset.pop('aspect_grouping', None)
        if 'evaluator' in dataset:
            iou_types = dataset['evaluator']['kwargs']['iou_types']
            if 'keypoints' not in iou_types:
                iou_types.append('keypoints')
    elif multi_task:
        raise NotImplementedError(f'{multi_task} not supported')


def L1_loss(second_stage):
    if second_stage is None:
        print('L1 loss is for second stage, which is not provided in you setting')
        return False
    else:
        second_stage['kwargs']['cfg']['smooth_l1_sigma'] = 1000
    return True


def filter_none(container):
    if isinstance(container, list) or isinstance(container, tuple):
        return [_ for _ in container if _]
    elif isinstance(container, dict):
        return OrderedDict({k: v for k, v in container.items() if v})
    else:
        return container


def yml_generate(args, dump=True):
    dataset = get_dataset(args.dataset, args.multi_task_stage, args.num_classes)
    args.num_classes = dataset['num_classes']
    trainer = get_trainer(dataset, args.IterMultiplier, args.multi_task_stage)
    saver = get_saver(args.backbone)
    fp16 = get_fp16(args.FP16)

    # Assume that a detector is a model-flow like backbone -> feature -> first stage -> second stage -> multi-task stage
    backbone = get_backbone(args.backbone, args.feature, args.first_stage)
    feature = get_feature(args.feature, args.first_stage)
    first_stage = get_first_stage(args.first_stage, args.feature, args.num_classes, dataset)
    if args.first_stage != 'RetinaNet':
        second_stage = get_second_stage(args.second_stage, args.num_classes, args.feature, args.backbone)
        multi_task_stage = get_multi_task(args.multi_task_stage, args.num_classes, args.feature)
    else:
        if args.second_stage or args.multi_task_stage:
            raise NotImplementedError('RetinaNet is not supposed to be cascade with rcnn')
        second_stage = None
        multi_task_stage = None

    update_dataset_output(dataset, args.multi_task_stage)

    second_stage_grid_flag(args.multi_task_stage, second_stage)

    tricks = []
    if args.DCNv2:
        # only part of backbone && part of second_stage head support dcn for now
        if dcnv2(backbone, second_stage):
            tricks.append('DCNv2')

    if args.SyncBN:
        if sync_bn(backbone, second_stage, args.multi_task_stage):
            tricks.append('SyncBN')

    if args.OHEM:
        # we only use OHEM in second stage
        if ohem(second_stage):
            tricks.append('OHEM')

    if args.L1:
        if L1_loss(second_stage):
            tricks.append('L1')

    if args.SoftNMS:
        if softnms(second_stage):
            tricks.append('SoftNMS')

    if args.MSTest:
        ms_test = multi_scale_test(dataset, second_stage, args.multi_task_stage)
        tricks.append('MSTest')
    else:
        ms_test = None

    if args.FP16:
        tricks.append('FP16')

    if args.IterMultiplier > 1:
        tricks.append(f'{args.IterMultiplier}x')

    detector = [backbone, feature, first_stage, second_stage, multi_task_stage]
    config = OrderedDict({
        'version': version,
        'dataset': dataset,
        'trainer': trainer,
        'saver': saver,
        'fp16': fp16,
        'ms_test': ms_test,
        'net': filter_none(detector)
    })
    config = filter_none(config)
    model = filter_none([args.backbone, args.feature, args.first_stage, args.second_stage, args.multi_task_stage])

    model_name = '-'.join(model + tricks + [args.dataset])

    output_yml = model_name + '.yml'
    if dump:
        with open(output_yml, 'w') as fout:
            ordered_yaml_dump(config, fout)
    return config, output_yml


def benchmark():

    # RFCN
    yml_generate(Args(backbone='resnet50', feature='C5', first_stage='RPN', second_stage='RFCN', dataset='VOC'))

    # FasterRCNN-FPN
    yml_generate(Args(backbone='mobilenetv2', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True))
    yml_generate(Args(backbone='shufflenetv2', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True))
    yml_generate(Args(backbone='resnet50', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True))
    yml_generate(
        Args(backbone='resnet50', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, IterMultiplier=2))
    yml_generate(Args(backbone='resnet101', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True))
    yml_generate(
        Args(backbone='resnet101', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, IterMultiplier=2))
    yml_generate(Args(backbone='resnext101_64x4d', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True))
    yml_generate(
        Args(
            backbone='resnext101_64x4d',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            FP16=True,
            IterMultiplier=2))

    # FasterRCNN-C4
    yml_generate(Args(backbone='resnet50', feature='C4', first_stage='RPN', second_stage='Res5', FP16=True))
    yml_generate(
        Args(backbone='resnet50', feature='C4', first_stage='RPN', second_stage='Res5', FP16=True, IterMultiplier=2))

    # FasterRCNN-C5
    yml_generate(Args(backbone='resnet50', feature='C5', first_stage='RPN', second_stage='FC', FP16=True))
    yml_generate(
        Args(backbone='resnet50', feature='C5', first_stage='RPN', second_stage='FC', FP16=True, IterMultiplier=2))

    # MaskRCNN-C4
    yml_generate(
        Args(
            backbone='resnet50',
            feature='C4',
            first_stage='RPN',
            second_stage='Res5',
            multi_task_stage='Mask',
            FP16=True))

    # MaskRCNN-FPN
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True))
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True,
            IterMultiplier=2))
    yml_generate(
        Args(
            backbone='resnet101',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True))
    yml_generate(
        Args(
            backbone='resnet101',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True,
            IterMultiplier=2))
    yml_generate(
        Args(
            backbone='resnext101_64x4d',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True))
    yml_generate(
        Args(
            backbone='resnext101_64x4d',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True,
            IterMultiplier=2))

    # CascadeRCNN
    yml_generate(
        Args(backbone='resnet50', feature='FPN', first_stage='RPN', second_stage='Cascade', FP16=True, SyncBN=True))
    yml_generate(
        Args(backbone='resnet101', feature='FPN', first_stage='RPN', second_stage='Cascade', FP16=True, SyncBN=True))

    # CascadeMaskRCNN
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='Cascade',
            multi_task_stage='Mask',
            FP16=True,
            SyncBN=True))
    yml_generate(
        Args(
            backbone='resnet101',
            feature='FPN',
            first_stage='RPN',
            second_stage='Cascade',
            multi_task_stage='Mask',
            FP16=True,
            SyncBN=True))

    # GridRCNN is a two stage detector, but it's implemented as a three stage detector.
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Grid',
            FP16=True,
            SyncBN=True,
            IterMultiplier=2))

    # RetinaNet:
    yml_generate(Args(backbone='resnet50', feature='FPN', first_stage='RetinaNet', FP16=True))
    yml_generate(Args(backbone='resnet50', feature='FPN', first_stage='RetinaNet', FP16=True, IterMultiplier=2))
    yml_generate(Args(backbone='resnet101', feature='FPN', first_stage='RetinaNet', FP16=True))
    yml_generate(Args(backbone='resnet101', feature='FPN', first_stage='RetinaNet', FP16=True, IterMultiplier=2))

    # GHM
    yml_generate(Args(backbone='resnet50', feature='FPN', first_stage='RetinaNet', FP16=True, GHM=True))

    # DCNv2
    # ResNet50-FasterRCNN-DCNv2
    yml_generate(Args(backbone='resnet50', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, DCNv2=True))
    yml_generate(
        Args(backbone='resnext101_64x4d', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, DCNv2=True))
    # R50-MaskRCNN-DCNv2
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True,
            DCNv2=True))

    # L1 loss
    yml_generate(Args(backbone='resnet50', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, L1=True))
    yml_generate(
        Args(backbone='resnext101_64x4d', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, L1=True))
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True,
            L1=True))

    # OHEM
    yml_generate(Args(backbone='resnet50', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, OHEM=True))
    yml_generate(
        Args(backbone='resnext101_64x4d', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, OHEM=True))
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Mask',
            FP16=True,
            OHEM=True))

    # SoftNMS
    yml_generate(
        Args(backbone='resnet50', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, SoftNMS=True))

    # MSTest
    yml_generate(Args(backbone='resnet50', feature='FPN', first_stage='RPN', second_stage='FC', FP16=True, MSTest=True))
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='Cascade',
            multi_task_stage='Mask',
            FP16=True,
            SyncBN=True,
            MSTest=True))
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='FC',
            multi_task_stage='Grid',
            FP16=True,
            SyncBN=True,
            MSTest=True))

    # Cascade + DCNv2 + L1 + OHEM + SoftNMS + MSTest
    yml_generate(
        Args(
            backbone='resnet50',
            feature='FPN',
            first_stage='RPN',
            second_stage='Cascade',
            FP16=True,
            DCNv2=True,
            L1=True,
            OHEM=True,
            SoftNMS=True,
            MSTest=True))
