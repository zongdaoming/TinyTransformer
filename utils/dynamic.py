#
# Copyright (c) 2021 SenseTime. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import importlib
from pathlib import Path
import logging
from collections import OrderedDict
import argparse
import os
import json
import torch
import logging


def filter_modules(model_state_dict, modules):
    """Filter non-matched modules in module_state_dict.
    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer
    Return:
        new_mods (list): the update module list
    """
    new_mods = []
    incorrect_mods = []

    mods_model = list(model_state_dict.keys())
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]

    if incorrect_mods:
        print(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        print("for information, the existing modules in model are:")
        print("%s", mods_model)

    return new_mods


def transfer_verification(model_state_dict, partial_state_dict, modules):
    """Verify tuples (key, shape) for input model modules match specified modules.
    Args:
        model_state_dict (OrderedDict): the initial model state_dict
        partial_state_dict (OrderedDict): the trained model state_dict
        modules (list): specified module list for transfer
    Return:
        (boolean): allow transfer
    """
    modules_model = []
    partial_modules = []

    for key_p, value_p in partial_state_dict.items():
        if any(key_p.startswith(m) for m in modules):
            partial_modules += [(key_p, value_p.shape)]

    for key_m, value_m in model_state_dict.items():
        if any(key_m.startswith(m) for m in modules):
            modules_model += [(key_m, value_m.shape)]

    len_match = len(modules_model) == len(partial_modules)

    module_match = sorted(modules_model, key=lambda x: (x[0], x[1])) == sorted(
        partial_modules, key=lambda x: (x[0], x[1])
    )

    return len_match and module_match


def get_partial_state_dict(model_state_dict, modules):
    """Create state_dict with specified modules matching input model modules.
    Note that get_partial_lm_state_dict is used if a LM specified.
    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer
    Return:
        new_state_dict (OrderedDict): the updated state_dict
    """
    new_state_dict = OrderedDict()

    for key, value in model_state_dict.items():
        if any(key.startswith(m) for m in modules):
            new_state_dict[key] = value

    return new_state_dict

def print_new_keys(state_dict, modules, model_path):
    print("loading %s from model: %s", modules, model_path)
    for k in state_dict.keys():
        print("override %s" % k)

def load_trained_modules(model_path, modules, model_without_ddp):
    """
    :param  model_path (str): args.checkpoint
    :param  modules (list): specified module list for transfer
    :return model (torch.nn.Module): The model with pretrained modules.
    """
    
    main_state_dict = model_without_ddp.state_dict()
    if model_path is not None:
        if os.path.isfile(model_path):
            model_state_dict = torch.load(model_path, map_location='cpu')
            model_state_dict = model_state_dict['model']
            modules = filter_modules(model_state_dict, modules)
            partial_state_dict = get_partial_state_dict(model_state_dict, modules)
            if partial_state_dict:
                if transfer_verification(
                    main_state_dict, partial_state_dict, modules
                ):
                    print_new_keys(partial_state_dict, modules, model_path)
                    main_state_dict.update(partial_state_dict)
                else:
                    print(
                        f"modules {modules} in model {model_path} "
                        f"don't match your training config",
                    )
        else:
            print("model was not found : %s", model_path)
    model_without_ddp.load_state_dict(main_state_dict)
    return model_without_ddp
