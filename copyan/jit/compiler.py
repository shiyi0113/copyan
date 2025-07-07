# --- Start of new, correct content ---

import hashlib
import functools
import os
import re
import subprocess
import uuid
import platform
import shutil
import torch
from torch.utils.cpp_extension import CUDA_HOME

from .runtime import Runtime, RuntimeCache
from .template import typename_map

runtime_cache = RuntimeCache()
IS_WINDOWS = platform.system() == 'Windows'

def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]

@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    return os.path.normpath(f'{os.path.dirname(os.path.abspath(__file__))}/../include')

@functools.lru_cache(maxsize=None)
def get_nvcc_compiler():
    if CUDA_HOME:
        nvcc_path = os.path.join(CUDA_HOME, 'bin', 'nvcc')
        if IS_WINDOWS:
            nvcc_path += '.exe'
        if os.path.exists(nvcc_path):
            return nvcc_path, "12.3"
    raise RuntimeError('Cannot find NVCC. Please set CUDA_HOME environment variable.')

@functools.lru_cache(maxsize=None)
def get_cache_dir():
    path = os.path.expanduser('~/.cache/copyan')
    os.makedirs(path, exist_ok=True)
    return path

def build(name: str, arg_defs: tuple, code: str) -> Runtime:
    lib_ext = '.dll' if IS_WINDOWS else '.so'

    common_flags = ['-std=c++20', '-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda']

    major, minor = torch.cuda.get_device_capability()
    arch_code = f"{major}{minor}"

    arch_map = {
        '80': '-gencode=arch=compute_80,code=sm_80',
        '86': '-gencode=arch=compute_86,code=sm_86',
        '89': '-gencode=arch=compute_89,code=sm_89',
        '90': '-gencode=arch=compute_90a,code=sm_90a',
        '100': '-gencode=arch=compute_100,code=sm_100',
        '120': '-gencode=arch=compute_100,code=sm_100', 
    }

    if arch_code in arch_map:
        print(f"Detected local GPU sm_{arch_code}, adding gencode flag: {arch_map[arch_code]}")
        common_flags.append(arch_map[arch_code])
    else:
        print(f"Warning: Unsupported GPU sm_{arch_code}. Falling back to sm_90a.")
        common_flags.append('-gencode=arch=compute_90a,code=sm_90a')

    if IS_WINDOWS:
        flags = [*common_flags, '-shared', '-Xcompiler=/MD']
    else:
        flags = [*common_flags, '-shared', '-Xcompiler', '-fPIC']

    include_dirs = [get_jit_include_dir()]

    signature = f'{name}$${code}$${flags}'
    kernel_name = f'kernel.{name}.{hash_to_hex(signature)}'
    path = os.path.join(get_cache_dir(), kernel_name)

    if runtime_cache[path] is not None:
        return runtime_cache[path]

    os.makedirs(path, exist_ok=True)

    src_path = os.path.join(path, 'kernel.cu')
    with open(src_path, 'w') as f:
        f.write(code)

    lib_path = os.path.join(path, f'kernel{lib_ext}')

    command = [get_nvcc_compiler()[0], src_path, '-o', lib_path, *flags, *[f'-I{d}' for d in include_dirs]]

    print(f"Compiling with command: {' '.join(command)}")

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Compilation failed: {e}')

    # --- 关键修正 ---
    # 检查 arg_defs 是否为空
    with open(os.path.join(path, 'kernel.args'), 'w') as f:
        if not arg_defs:
            # 如果没有参数，写入一个合法的空元组字符串
            f.write("()")
        else:
            # 如果有参数，正常生成字符串
            args_str = ', '.join([f"('{arg_name}', {typename_map[arg_type]})" for arg_name, arg_type in arg_defs])
            f.write(f"({args_str},)")

    runtime_cache[path] = Runtime(path)
    return runtime_cache[path]