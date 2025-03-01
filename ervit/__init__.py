# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.eventrep_semantic import (
    EventRepSemanticDatasetMapper
)

# models
from .ERViT_MaskFormer import ERViT_MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA
