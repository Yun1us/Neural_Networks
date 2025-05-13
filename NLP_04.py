import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, gradcheck

#autograd custom
class MyMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x

class MyMax(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.maximum(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        maskx = x > y
        masky = x < y
        maskequal = x == y
        grad_x = torch.where(maskx, grad_output, torch.where(maskequal, grad_output * 0.5, torch.zeros_like(grad_output)))
        grad_y = torch.where(masky, grad_output, torch.where(maskequal, grad_output * 0.5, torch.zeros_like(grad_output)))
        return grad_x, grad_y

class MyCos(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.cos(x)

    @staticmethod
    def backward(ctx, gradient_output):
        x, = ctx.saved_tensors
        return gradient_output * -torch.sin(x)

#gradcheck
x = torch.randn(3, dtype=torch.double, requires_grad=True)
y = torch.randn(3, dtype=torch.double, requires_grad=True)
print(gradcheck(MyMul.apply, (x, y)))
print(gradcheck(MyMax.apply, (x, y)))



class CosLinear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight_with_cos = MyCos.apply(self.weight)
        return F.linear(input, weight_with_cos, self.bias)


#Neural Network class 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(64, 32)
        self.hidden_layer = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, input):
        outputs = []
        if self.training:
            x1 = F.relu(self.linear(input))
            num_layers = torch.randint(1, 5, (1,)).item()
            for _ in range(num_layers):
                x1 = F.relu(self.hidden_layer(x1))
            return self.output_layer(x1)
        else:
            for n in range(1, 5):
                x = F.relu(self.linear(input))
                for _ in range(n):
                    x = F.relu(self.hidden_layer(x))
                out = self.output_layer(x)
                outputs.append(out)
            return torch.stack(outputs).mean(dim=0)


#Numpy Dataset 
n = 2**14
dim_input = 64
dim_output = 1
X = np.random.randn(n, dim_input).astype(np.float32)
true_weights = np.random.randn(dim_input, dim_output)
y = X @ true_weights + np.random.randn(n, dim_output) * 0.1

#Train / Test split 
samples = np.random.permutation(n)
split = int(n * 0.75)
train = samples[:split]
test = samples[split:]
X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]

#loss and model init
model = Net()
loss_CE = nn.CrossEntropyLoss()

# Train loop anfang
epoch = 5
for e in range(epoch):
    train_sum_loss = 0.0
    validation_sum_loss = 0.0
    model.train()
