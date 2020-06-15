from enum import Enum

import numpy as np
import tensorflow as tf

import config
import utils.data as data_utils
import utils.shuffle as shuffle_utils
from layers.shuffle import LinearTransform, QuaternarySwitchUnit

