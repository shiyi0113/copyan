# 内容复制到 my_yan/my_yan/jit/runtime.py

import ctypes
import os
import platform
import torch
from typing import Optional

from .template import map_ctype

IS_WINDOWS = platform.system() == "Windows"

class Runtime:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.args = None
        
    def __call__(self, *args) -> int:
        if self.lib is None or self.args is None:
            lib_name = os.path.join(self.path, "kernel.dll" if IS_WINDOWS else "kernel.so")
            self.lib = ctypes.CDLL(lib_name)
            with open(os.path.join(self.path, "kernel.args"), "r") as f:
                self.args = eval(f.read())

        cargs = [map_ctype(arg) for arg in args]
        
        return_code = ctypes.c_int(0)
        self.lib.launch(*cargs, ctypes.byref(return_code))
        return return_code.value

class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __getitem__(self, path: str) -> Optional[Runtime]:
        lib_name = os.path.join(path, "kernel.dll" if IS_WINDOWS else "kernel.so")
        if path in self.cache:
            return self.cache[path]
        if os.path.exists(lib_name):
            runtime = Runtime(path)
            self.cache[path] = runtime
            return runtime
        return None

    def __setitem__(self, path, runtime) -> None:
        self.cache[path] = runtime