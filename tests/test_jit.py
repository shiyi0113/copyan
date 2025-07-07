import torch

# 导入我们新创建的 JIT 接口
from copyan.jit_kernels import null_op

def test_jit_framework():
    """
    测试 JIT 框架能否成功编译并调用一个空操作 Kernel。
    """
    print("Testing JIT Framework...")
    print("This test will try to compile and run a do-nothing CUDA kernel.")
    
    try:
        # 调用我们的空操作函数
        # 如果 JIT 框架的任何一个环节（代码生成、编译、加载、调用）出错，
        #这里就会抛出异常。
        return_code = null_op()

        # 检查 C++ wrapper 返回的错误码
        if return_code == 0:
            print("\nJIT Framework Test PASSED! ✅")
            print("Successfully compiled and invoked a CUDA kernel.")
        else:
            print(f"\nJIT Framework Test FAILED! ❌")
            print(f"Kernel invocation returned a non-zero error code: {return_code}")

    except Exception as e:
        print(f"\nJIT Framework Test FAILED! ❌")
        print(f"An exception occurred during the process: {e}")

if __name__ == "__main__":
    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
    else:
        test_jit_framework()