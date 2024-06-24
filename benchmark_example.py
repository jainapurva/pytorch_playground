import torch
import copy
from torchao.utils import benchmark_model
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# params
device = "cpu"  # Options : cpu, cuda, mps
dtype = torch.bfloat16
m = ToyLinearModel(1024, 1024, 1024).eval().to(dtype).to(device)
m_bf16 = copy.deepcopy(m)
example_inputs = m.example_inputs(dtype=dtype, device=device)
m_bf16 = torch.compile(m_bf16, mode='max-autotune')
num_runs = 100
bf16_time = benchmark_model(m_bf16, num_runs, example_inputs[0])
print(f"bf16 mean time: {bf16_time} on: {device} device.")
