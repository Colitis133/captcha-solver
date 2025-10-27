"""Quick smoke test to ensure modules import and model forward pass."""
def run():
    import torch
    from .model import make_model
    m = make_model(36 + 1)
    x = torch.randn(1, 1, 60, 160)
    y = m(x)
    print('smoke ok, out shape', y.shape)


if __name__ == '__main__':
    run()
