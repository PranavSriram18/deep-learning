import torch 

def random_dot(n: int):
    x = torch.randn(n)
    x /= x.norm()
    y = torch.randn(n)
    y /= y.norm()
    print(f"Result for n = {n}: {x @ y}")

if __name__=="__main__":
    for i in range(5, 9):
        random_dot(10 ** i)
