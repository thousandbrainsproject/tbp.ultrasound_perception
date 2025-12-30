# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import importlib.util
import os
import sys


def import_config_from_monty(config_file, config_name):
    """Import config from tbp.monty.

    NOTE: This looks at a local copy of the defualts that used to be found in the
    tbp.monty directory.
    TODO: Remove this as part of movement to using new Hydra configs.
    """
    # Get the directory of the current file (config_utils.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, "tbp_monty_pre_hydra_configs", config_file)

    # Create proper module name for package structure
    module_name = (
        f"configs.tbp_monty_pre_hydra_configs.{os.path.splitext(config_file)[0]}"
    )

    spec = importlib.util.spec_from_file_location(module_name, full_path)
    module = importlib.util.module_from_spec(spec)

    # Add to sys.modules so relative imports work
    sys.modules[module_name] = module

    spec.loader.exec_module(module)
    return getattr(module, config_name)
