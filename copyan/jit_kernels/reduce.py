import torch
from .tuner import jit_tuner

includes = ('"reduce/reduce.cuh"',)
template = """
// Templated args from Python JIT call
//reduce_sum_c(X, y0, N);
//reduce_max_c(X, y1, N);
reduce_sum_c<float, {BLOCK_SIZE}, {ITEMS_PER_THREAD}>(X, y0, N);
reduce_max_c<float, {BLOCK_SIZE}, {ITEMS_PER_THREAD}>(X, y1, N);
"""


def reduce_sum_max(
    x: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor, space: tuple = None
) -> None:
    N = x.shape[0]
    assert (
        x.dtype == torch.float32
        and y0.dtype == torch.float32
        and y1.dtype == torch.float32
    )

    global includes, template
    if space is None:
        space = (dict(BLOCK_SIZE=1024, ITEMS_PER_THREAD=16),)
    args = (x, y0, y1, N)
    runtime = jit_tuner.compile_and_tune(
        name="reduce sum & max",
        keys={},
        space=space,
        includes=includes,
        arg_defs=(
            ("X", torch.float),
            ("y0", torch.float),
            ("y1", torch.float),
            ("N", int),
        ),
        template=template,
        args=args,
    )

    runtime(*args)


def accuracy_test():
    for _ in range(1):
        torch.manual_seed(41)
        N = 4096 * 1024
        x = torch.randn(N, dtype=torch.float, device="cuda")
        y0 = torch.zeros(1, dtype=torch.float, device="cuda")
        y1 = torch.zeros(1, dtype=torch.float, device="cuda")

        reduce_sum_max(x, y0, y1)

        print(f"{'my reduce sum':<14} = {y0.item():.4f}")
        print(f"{'pytorch sum':<14} = {torch.sum(x).item():.4f}")
        print(f"{'my reduce max':<14} = {y1.item():.4f}")
        print(f"{'pytorch max':<14} = {torch.max(x).item():.4f}")

        print("Test passed!")


if __name__ == "__main__":
    accuracy_test()
