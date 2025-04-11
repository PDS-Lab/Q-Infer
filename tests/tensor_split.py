import time
import torch
import torch.nn as nn

batch_size = 64
print_flag = False
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用 GPU 0

class DataStatistics:
    def __init__(self):
        # 存储数据，键是名字，值是包含所有插入数的列表
        self.data = {}

    def add_data(self, name, value):
        """添加数据并自动更新平均值"""
        if name not in self.data:
            self.data[name] = []

        self.data[name].append(value)
    
    def get_average(self, name: str) -> float:
        """获取某个名字的平均值"""
        if name not in self.data or len(self.data[name]) == 0:
            raise ValueError(f"'{name}' 的数据不存在或为空")

        return sum(self.data[name]) / len(self.data[name])

    def __str__(self):
        """输出所有数据及其平均值"""
        result = ""
        for name, values in self.data.items():
            avg = self.get_average(name)
            result += f"{name}: avg={avg:.2f}\n"
        return result

g_stat = DataStatistics()

class TensorParallelLayer(nn.Module):
    def __init__(self, hidden_size, neuron, tensor_split, batch_split):
        super(TensorParallelLayer, self).__init__()
        
        # 在不同设备上初始化参数，切分hidden维度
        self.linear = nn.Linear(hidden_size, neuron);
        self.linear_cpu = nn.Linear(hidden_size, neuron - tensor_split);
        self.linear_gpu = nn.Linear(hidden_size, tensor_split);

        self.cpu_batch = batch_size - batch_split
        self.gpu_batch = batch_split

    def forward_tensor_data_split(self, x):
        # 将输入在hidden维度上切分
        global g_stat
        start = time.perf_counter()
        x_cpu, x_gpu = torch.split(x, [self.cpu_batch, self.gpu_batch], dim=0)

        # 在不同设备上计算
        gpu_tensor = self.linear_gpu.to('cuda')
        torch.cuda.synchronize()
        p1 = time.perf_counter()
        g_stat.add_data("move to GPU time", (p1 - start) * 1e6)

        out_gpu = gpu_tensor(x_gpu)  # GPU计算
        torch.cuda.synchronize()
        p2 = time.perf_counter()
        g_stat.add_data("GPU compute time", (p2 - p1) * 1e6)

        x_cpu = x_cpu.to('cpu')
        x_gpu = x_gpu.clone().to('cpu')
        torch.cuda.synchronize()
        p3 = time.perf_counter()
        g_stat.add_data("activation to CPU", (p3 - p2) * 1e6)

        out_cpu1 = self.linear(x_cpu)  # CPU计算
        p4 = time.perf_counter()
        g_stat.add_data("CPU compute time 1", (p4 - p3) * 1e6)

        out_cpu2 = self.linear_cpu(x_gpu)
        p5 = time.perf_counter()
        g_stat.add_data("CPU compute time 2", (p5 - p4) * 1e6)

        # 将CPU的结果移动到GPU并合并
        out_cpu1 = out_cpu1.to('cuda')
        out_cpu2 = out_cpu2.to('cuda')
        torch.cuda.synchronize()
        p6 = time.perf_counter()
        g_stat.add_data("activation to GPU", (p6 - p5) * 1e6)

        output2 = torch.concat((out_cpu2, out_gpu), dim=1)
        output  = torch.concat((out_cpu1, output2), dim=0)
        torch.cuda.synchronize()
        p7 = time.perf_counter()
        g_stat.add_data("concate", (p7 - p6) * 1e6)

        return output
    
    def forward_tensor_split(self, x):
        # 将输入在hidden维度上切分
        global g_stat
        start = time.perf_counter()
        # 在不同设备上计算
        gpu_tensor = self.linear_gpu.to('cuda')
        torch.cuda.synchronize()
        p1 = time.perf_counter()
        g_stat.add_data("move to GPU time", (p1 - start) * 1e6)


        out_gpu = gpu_tensor(x)  # GPU计算
        torch.cuda.synchronize()
        p2 = time.perf_counter()
        g_stat.add_data("GPU compute time", (p2 - p1) * 1e6)

        x_cpu = x.clone().to('cpu')
        torch.cuda.synchronize()
        p3 = time.perf_counter()
        g_stat.add_data("activation to CPU", (p3 - p2) * 1e6)

        out_cpu = self.linear_cpu(x_cpu)
        p5 = time.perf_counter()
        g_stat.add_data("CPU compute time", (p5 - p3) * 1e6)

        # 将CPU的结果移动到GPU并合并
        out_cpu = out_cpu.to('cuda')
        torch.cuda.synchronize()
        p6 = time.perf_counter()
        g_stat.add_data("activation to GPU", (p6 - p5) * 1e6)

        output = torch.concat((out_cpu, out_gpu), dim=1)
        torch.cuda.synchronize()
        p7 = time.perf_counter()
        g_stat.add_data("concate", (p7 - p6) * 1e6)

        return output
    
    def forward(self, x):
        if self.cpu_batch != 0:
            output = self.forward_tensor_data_split(x)
        else:
            output = self.forward_tensor_split(x)
        return output
    
class TensorParallelNetwork(nn.Module):
    def __init__(self, hidden_size, neuron, tensor_split, batch_split, num_layers):
        super(TensorParallelNetwork, self).__init__()
        self.down = nn.Linear(neuron, hidden_size).to('cuda')
        # 创建多层张量并行网络

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TensorParallelLayer(hidden_size, neuron, tensor_split, batch_split))
            self.layers.append(self.down)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        # 依次通过每一层
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
import sys


# 测试网络
if __name__ == "__main__":
    batch_size, hidden_size, neuron, num_layers = 128, 8192, 32768, 16
    tensor_split = 4096
    batch_split = 96

    if len(sys.argv) != 4:
        print("<cmd> <batch_size> <batch_split> <tensor_split>")
        exit(-1)

    batch_size = int(sys.argv[1])
    batch_split = int(sys.argv[2])
    tensor_split = int(sys.argv[3])

    # batch切分 + 参数切分
    model = TensorParallelNetwork(hidden_size, neuron, tensor_split, batch_split, num_layers)

    # 执行前向传播
    # 创建随机输入，假设输入张量在GPU上
    g_stat = DataStatistics()
    X = torch.randn(batch_size, hidden_size).to("cuda")
    print_flag = True
    start = time.perf_counter()
    output = model(X)
    p = time.perf_counter()
    print(f"Batch-Tensor Split total time: {(p - start) * 1e6:.3f}us")
    print(f"GPU :[{batch_split}, {hidden_size}] * [{hidden_size}, {tensor_split}]")
    print(f"CPU :[{batch_size - batch_split}, {hidden_size}] * [{hidden_size}, {neuron}]")
    print(f"CPU :[{batch_split}, {hidden_size}] * [{hidden_size}, {neuron - tensor_split}]")
    print(g_stat)

    # 参数切分
    g_stat = DataStatistics()
    tensor_split = batch_split * tensor_split // batch_size
    model2 = TensorParallelNetwork(hidden_size, neuron, tensor_split, batch_size, num_layers)
    X = torch.randn(batch_size, hidden_size).to("cuda")
    print_flag = True
    start = time.perf_counter()
    output = model2(X)
    p = time.perf_counter()
    print(f"Tensor Split total time: {(p - start) * 1e6:.3f}us")
    print(f"GPU :[{batch_size}, {hidden_size}] * [{hidden_size}, {tensor_split}]")
    print(f"CPU :[{batch_size}, {hidden_size}] * [{hidden_size}, {neuron - tensor_split}]")
    print(g_stat)