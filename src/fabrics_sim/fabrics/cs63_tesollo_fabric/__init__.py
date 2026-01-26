# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# This package implements the CS63 Tesollo fabric with multiple variants.
# By naming this folder 'cs63_tesollo_fabric', it seamlessly replaces the original 
# 'cs63_tesollo_fabric.py' file while providing a modular internal structure.

from .force_variant import CS63TesolloForceFabric
from .attractor_variant import (
    _CS63TesolloGripperAttractorBase,
    CS63TesolloGripperLatentAttractorFabric,
    CS63TesolloGripperJointAttractorFabric
)
from .fingertip_attractor_variant import CS63TesolloFingertipAttractorFabric

# Compatibility aliases
CS63TesolloFabric = CS63TesolloFingertipAttractorFabric
