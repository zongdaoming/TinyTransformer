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
def dynamic_load_trained_modules(args,moduels,model_without_ddp):
    """
    :param input_dim:
    :param output_dim:
    :param args arguments
    """
    model_pth = args.checkpoint
    module = importlib.import_module('utils.dynamic')
    return module.load_trained_modules(model_pth,moduels,model_without_ddp)


