# set gradient to zero
import torch

weights = torch.ones(4, requires_grad=True)
print(weights)

for i in range(1):
    z = (weights*3).mean()
    print(z)
    z.backward()
    print(weights.grad)
    weights.grad.zero_()