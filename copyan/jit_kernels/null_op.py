from .tuner import jit_tuner

# 包含我们新创建的头文件
includes = ('"null_op.cuh"',)

# 定义 Kernel 的启动模板
template = """
    // 直接启动我们的空操作 Kernel
    // 它不需要任何参数
    null_op_kernel<<<1, 1>>>();
"""

def null_op():
    """
    这个函数会调用 JIT 编译器来构建和运行一个什么都不做的 Kernel。
    它的目的是测试整个编译和调用流程是否通畅。
    """
    # 这个 Kernel 不需要任何参数
    args = ()
    arg_defs = ()

    # 调用 JIT 编译器
    runtime = jit_tuner.compile_and_tune(
        name='null_op',
        keys={},
        space=(),
        includes=includes,
        arg_defs=arg_defs,
        template=template,
        args=args
    )

    # 调用编译好的函数
    return_code = runtime(*args)
    return return_code