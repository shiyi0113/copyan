import torch
import copyan
from copyan import bench_kineto


def test_sum_reduce():
    print("Testing reduce sum & max:")

    def test_func():
        N = 1024 * 4096 * 128
        x = torch.randn(N, dtype=torch.float, device="cuda")
        y0 = torch.zeros(1, dtype=torch.float, device="cuda")
        y1 = torch.zeros(1, dtype=torch.float, device="cuda")
        space = (
            dict(BLOCK_SIZE=512, ITEMS_PER_THREAD=8),
            dict(BLOCK_SIZE=512, ITEMS_PER_THREAD=16),
            dict(BLOCK_SIZE=1024, ITEMS_PER_THREAD=8),
            dict(BLOCK_SIZE=1024, ITEMS_PER_THREAD=16),
        )
        copyan.jit_kernels.reduce_sum_max(x, y0, y1, space)

    t = bench_kineto(
        test_func, ("SumOp<float>", "MaxOp<float>"), suppress_kineto_output=True
    )
    for i, time in enumerate(t):
        print(f" > Performance {i}: {time * 1e6:4.0f} us")
    total_time = sum(t)
    print(f" > Total Performance: {total_time * 1e6:4.0f} us")


if __name__ == "__main__":
    copyan.jit_kernels.reduce_sum_max_accuracy_test()
    test_sum_reduce()
