def bench_kineto(fn, kernel_names, num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: str = None, barrier_comm_profiling: bool = False, flush_l2: bool = False):
    # 导入所需的模块：os 用于操作系统交互，sys 用于系统相关操作，torch 用于 PyTorch 功能，torch.distributed 用于分布式操作。
    # empty_suppress 类：一个空的上下文管理器，用于在不需要抑制输出时作为占位符。
    # suppress_stdout_stderr 类：一个上下文管理器，用于抑制标准输出和标准错误输出。它会重定向 stdout 和 stderr 到 /dev/null（或 Windows 上的 null 设备）。

    using_nsys = os.environ.get('COPYAN_NSYS_PROFILING', False)
    # 检查环境变量 'COPYAN_NSYS_PROFILING' 是否被设置。如果设置了，表示用户可能正在使用 NVIDIA NSys 外部分析器。

    # For some auto-tuning kernels with prints
    fn()
    # 首次调用传入的函数 `fn`。这通常用于预热 GPU，或者对于某些自调整内核，可能需要运行一次以触发内部的编译或打印信息。

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output and not using_nsys else empty_suppress
    # 根据 `suppress_kineto_output` 和 `using_nsys` 的值，选择要使用的上下文管理器。
    # 如果 `suppress_kineto_output` 为 True 且没有使用 NSys，则使用 `suppress_stdout_stderr` 来抑制 PyTorch Profiler 的输出。
    # 否则，使用 `empty_suppress`，不抑制输出。

    with suppress():
    # 进入所选的上下文管理器，用于控制输出的显示。
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) if not using_nsys else None
        # 定义 PyTorch Profiler 的调度器。如果不是使用 NSys，调度器会设置为：
        # wait=0：不等待，立即开始。
        # warmup=1：跳过第一个迭代作为预热，不进行记录。
        # active=1：对第二个迭代进行记录。
        # repeat=1：重复整个周期一次（总共运行两次，一次预热一次激活）。
        # 如果是使用 NSys，调度器设为 None。

        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) if not using_nsys else empty_suppress()
        # 初始化 PyTorch Profiler。它被配置为只记录 CUDA 活动。
        # 如果是使用 NSys，profiler 则是一个空的上下文管理器。

        with profiler:
        # 进入 PyTorch Profiler 的上下文管理器，开始性能数据收集。
            for i in range(2):
            # 循环两次。如果 `schedule` 配置为 `warmup=1, active=1`，则第一次循环是预热，第二次是实际记录。
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                # 如果 `barrier_comm_profiling` 为 True，表示在进行有屏障通信的基准测试。
                    lhs = torch.randn((4096, 4096), dtype=torch.float, device='cuda')
                    rhs = torch.randn((4096, 4096), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    # 执行一个大的 GEMM (通用矩阵乘法) 操作。这有助于在分布式设置中引入大量 GPU 工作负载，以减少 CPU 启动开销的影响。
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                    # 执行一个分布式 `all_reduce` 操作。这作为一个通信屏障，确保所有进程在继续之前都已同步。
                for _ in range(num_tests):
                # 内部循环，`fn` 会在此循环中运行 `num_tests` 次。
                    if flush_l2:
                    # 如果 `flush_l2` 为 True，表示在每次运行前需要刷新 GPU 的 L2 缓存。
                        torch.empty(int(128e6 // 4), dtype=torch.int, device='cuda').zero_()
                        # 创建一个大的（128MB）CUDA 张量并将其填充为零。这会强制将数据写入 GPU 内存，从而刷新 L2 缓存。
                    fn()
                    # 调用传入的函数 `fn`，这是实际要进行基准测试的代码。
                
                if not using_nsys:
                # 如果不是使用 NSys，则调用 profiler 的 `step()` 方法。
                    profiler.step()
                    # 对于 PyTorch Profiler，`step()` 会推进调度器，例如从 `warmup` 阶段进入 `active` 阶段。
    
    if using_nsys:
    # 如果检测到正在使用 NSys 进行分析。
        return 1
        # 返回 1，因为 NSys 会在外部收集性能数据，此函数内部不进行详细的 Kineto 解析。
    
    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    # 确保 `kernel_names` 是字符串或元组类型。
    is_tupled = isinstance(kernel_names, tuple)
    # 记录 `kernel_names` 是否是元组。
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=300).split('\n')
    # 获取分析器的平均键统计表，按 'cuda_time_total' 排序，并限制名称列宽度，然后将其分割成行。
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    # 将 `kernel_names` 统一转换为元组，方便后续迭代处理。
    assert all([isinstance(name, str) for name in kernel_names])
    # 确保 `kernel_names` 元组中的所有元素都是字符串。
    for name in kernel_names:
    # 遍历每个内核名称。
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table: {prof_lines}'
        # 验证每个内核名称在分析表中只出现一次，确保准确性。
        
    # Save chrome traces
    if trace_path is not None:
    # 如果 `trace_path` 不为空，则导出 Chrome 跟踪文件。
        profiler.export_chrome_trace(trace_path)
        # 将性能分析数据导出为 Chrome 浏览器可识别的跟踪文件，用于可视化分析。

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    # 定义时间单位及其对应的缩放因子（毫秒到秒，微秒到秒）。
    kernel_times = []
    # 初始化一个列表，用于存储提取的内核时间。
    for name in kernel_names:
    # 再次遍历每个内核名称。
        for line in prof_lines:
        # 遍历分析报告的每一行。
            if name in line:
            # 如果当前行包含内核名称。
                time_str = line.split()[-2]
                # 提取倒数第二个字段，它包含时间字符串（例如 "10.5us"）。
                for unit, scale in units.items():
                # 遍历时间单位。
                    if unit in time_str:
                    # 如果时间字符串中包含当前单位。
                        kernel_times.append(float(time_str.replace(unit, '')) / scale)
                        # 移除单位，将字符串转换为浮点数，并根据单位缩放因子转换为秒，然后添加到 `kernel_times` 列表中。
                        break
                        # 找到匹配的单位后，跳出内部循环。
                break
                # 找到匹配的行后，跳出外部循环。
    return tuple(kernel_times) if is_tupled else kernel_times[0]
    # 如果原始 `kernel_names` 是元组，则返回一个包含所有内核时间的元组；否则，返回第一个（也是唯一一个）内核时间。
