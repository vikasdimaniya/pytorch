import torch

x = torch.randn(4, requires_grad=True)
print(x)
y = torch.randn(4, requires_grad=True)
print(y)
z = x*y
print(z)

mean_tensor = z.mean()
print(mean_tensor)

mean_tensor.backward()
print(x.grad)
print(x)


print("--------------")
print(y.grad)
print(y)