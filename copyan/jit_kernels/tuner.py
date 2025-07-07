# 内容复制到 my_yan/my_yan/jit_kernels/tuner.py

from ..jit import build, cpp_format, generate

class JITTuner:
    def __init__(self):
        self.cache = {}

    def compile_and_tune(self, name, keys, space, includes, arg_defs, template, args):
        signature = (name, str(keys))
        if signature in self.cache:
            return self.cache[signature]
        
        # 简化：目前不进行自动调优，直接使用 keys 编译
        code = generate(includes, arg_defs, cpp_format(template, keys))
        runtime = build(name, arg_defs, code)
        
        self.cache[signature] = runtime
        return runtime

jit_tuner = JITTuner()