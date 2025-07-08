import torch

from . import jit
from .utils import bench_kineto, calc_diff
from .jit_kernels import reduce_sum_max_accuracy_test