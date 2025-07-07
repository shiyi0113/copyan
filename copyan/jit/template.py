# --- Start of final, correct content ---

import ctypes
import torch
from typing import Any, Dict

# 类型映射 (保持不变)
typename_map: Dict[Any, str] = {
    bool: "bool", int: "int", float: "float",
    torch.int32: "torch.int32", torch.float32: "torch.float32",
    torch.float16: "torch.half", torch.bfloat16: "torch.bfloat16",
    torch.cuda.Stream: "torch.cuda.Stream",
}

ctype_map: Dict[Any, Any] = {
    bool: ctypes.c_bool, int: ctypes.c_int, float: ctypes.c_float,
    torch.int32: ctypes.c_void_p, torch.float32: ctypes.c_void_p,
    torch.half: ctypes.c_void_p, torch.bfloat16: ctypes.c_void_p,
    torch.cuda.Stream: ctypes.c_void_p,
}

genc_map = {
    bool: ("bool", "bool"), int: ("int", "int"), float: ("float", "float"),
    torch.int32: ("void*", "int*"), torch.float32: ("void*", "float*"),
    torch.half: ("void*", "half*"), torch.bfloat16: ("void*", "__nv_bfloat16*"),
    torch.cuda.Stream: ("void*", "cudaStream_t"),
}

def map_ctype(value: Any) -> Any:
    v_type = value.dtype if isinstance(value, torch.Tensor) else type(value)
    ctype = ctype_map.get(v_type)
    if ctype is None:
        raise TypeError(f"Unsupported type for ctype mapping: {v_type}")

    if isinstance(value, torch.Tensor):
        return ctype(value.data_ptr())
    if isinstance(value, torch.cuda.Stream):
        return ctype(value.cuda_stream)
    return ctype(value)

def cpp_format(template: str, keys: Dict[str, Any]) -> str:
    for key, value in keys.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template

def generate(includes: tuple, arg_defs: tuple, body: str) -> str:
    # 添加头文件
    code = '#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n'
    for include in includes:
        code += f'#include {include}\n'

    # 定义可从 Python 调用的 C 风格启动函数
    code += '\nextern "C" void launch('

    # 列出函数参数
    raw_args = []
    for name, dtype in arg_defs:
        raw_args.append(f"{genc_map[dtype][0]} raw_{name}")
    raw_args.append("int& __return_code")
    code += ", ".join(raw_args)
    code += ") {\n"

    # 类型转换
    code += "    // Cast raw types to CUDA types\n"
    for arg_name, arg_type in arg_defs:
        if genc_map[arg_type][0] != genc_map[arg_type][1]:
            code += f"    auto {arg_name} = reinterpret_cast<{genc_map[arg_type][1]}>(raw_{arg_name});\n"

    # --- 关键改动：移除了自动替换变量名的逻辑 ---
    # 直接插入用户提供的、已经包含正确变量名(raw_M 等)的 body
    code += "\n    // User-provided kernel launch logic\n"
    body_indented = "\n".join([("    " + line) for line in body.split("\n")])
    code += body_indented

    # 设置返回码并关闭函数
    code += "\n    __return_code = 0;\n"
    code += "}\n"

    return code