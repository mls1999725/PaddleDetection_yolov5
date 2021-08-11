import paddle
import torch

import numpy as np

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def reset_parameters(pmodel, tmodel, data=None):
    'reset parameters'
    sublayers = [(n, m) for n, m in pmodel.named_sublayers(include_self=True)]
    submodules = [(n, m) for n, m in tmodel.named_modules()]

    print(len(sublayers), len(submodules))
    assert len(sublayers) == len(submodules)

    for i, (n, m) in enumerate(sublayers):
        
        _m = submodules[i][1]

        if isinstance(m, paddle.nn.Conv2D):
            m.weight.set_value(_m.weight.data.numpy())
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.set_value(_m.bias.data.numpy())

        elif isinstance(m, paddle.nn.BatchNorm2D):
            m.weight.set_value(_m.weight.data.numpy())
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.set_value(_m.bias.data.numpy())

            m._mean.set_value(_m.running_mean.data.numpy())
            m._variance.set_value(_m.running_var.data.numpy())

    if data is None:
        data = np.random.randn(1,3,20,20).astype(np.float32)

    tdata = torch.tensor(data[...])
    pdata = paddle.to_tensor(data[...])
    tdata.requires_grad = True
    pdata.stop_gradient = False

    tout = tmodel(tdata)
    pout = pmodel(pdata)

    print('---------forward---------')
    print('torch', tout.mean(), tout.sum())
    print('paddle', pout.mean(), pout.sum())
    np.testing.assert_almost_equal(pout.numpy(), tout.data.numpy(), decimal=5)

    print('---------backward---------')
    tout.sum().backward()
    pout.sum().backward()

    print('torch', tdata.grad.mean(), tdata.grad.sum())
    print('paddle', pdata.grad.mean(), pdata.grad.sum())
    np.testing.assert_almost_equal(pdata.grad.numpy(), tdata.grad.data.numpy(), decimal=5)

    check_parameters(pmodel, tmodel)
    print('-------<after checking forward, backward for input and parameters>----------')
    print('----------------------------reset parameters done---------------------------')

def check_parameters(pmodel, tmodel, decimal=4):
    'check parameters'

    tnp = [(n, p) for n, p in tmodel.named_parameters() if p.requires_grad]
    pnp = [(n, p) for n, p in pmodel.named_parameters() if not p.stop_gradient]
    assert len(tnp) == len(pnp)

    for _tnp, _pnp in zip(tnp, pnp):
        _tp = _tnp[-1]; _pp = _pnp[-1]

        if list(_tp.shape) == list(_pp.shape):
            np.testing.assert_almost_equal(_tp.data.numpy(), _pp.numpy(), decimal=decimal)
            np.testing.assert_almost_equal(_tp.grad.data.numpy(), _pp.grad.numpy(), decimal=decimal)
        elif list(_tp.shape[::-1]) == list(_pp.shape):
            np.testing.assert_almost_equal(_tp.data.numpy().T, _pp.numpy(), decimal=decimal)
            np.testing.assert_almost_equal(_tp.grad.data.numpy().T, _pp.grad.numpy(), decimal=decimal)
        else:
            raise RuntimeError('--')

def get_torch_mm():
    class TorchConv(torch.nn.Module):
    # Standard convolution
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
            super().__init__()
            self.conv = torch.nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
            self.bn = torch.nn.BatchNorm2d(c2)
            self.act = torch.nn.SiLU() if act is True else (act if isinstance(act, torch.nn.Module) else torch.nn.Identity())

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

        def fuseforward(self, x):
            return self.act(self.conv(x))

    class TorchBottleneck(torch.nn.Module):
    # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = TorchConv(c1, c_, 1, 1)
            self.cv2 = TorchConv(c_, c2, 3, 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    class TorchC3(torch.nn.Module):
    # CSP Bottleneck with 3 convolutions
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = TorchConv(c1, c_, 1, 1)
            self.cv2 = TorchConv(c1, c_, 1, 1)
            self.cv3 = TorchConv(2 * c_, c2, 1)  # act=FReLU(c2)
            self.m = torch.nn.Sequential(*[TorchBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
            # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

        def forward(self, x):
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
    model = TorchC3(c1=3, c2=3, n=1)
    return model

def get_paddle_mm():
    class PaddleConv(paddle.nn.Layer):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super(PaddleConv, self).__init__()
            self.conv = paddle.nn.Conv2D(c1, c2, k, s, autopad(k, p),
                              groups=g, bias_attr=False)
            self.bn = paddle.nn.BatchNorm2D(c2, momentum=0.9, epsilon=1e-5)
            self.act = paddle.nn.Silu() if act is True else (
                act if isinstance(act, paddle.nn.Layer) else paddle.nn.Identity())

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

        def fuseforward(self, x):
            return self.act(self.conv(x))

    class PaddleBottleneck(paddle.nn.Layer):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, groups, expansion
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
            super(PaddleBottleneck, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = PaddleConv(c1, c_, 1, 1)
            self.cv2 = PaddleConv(c_, c2, 3, 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
   
    class PaddleC3(paddle.nn.Layer):
    # CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super(PaddleC3, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = PaddleConv(c1, c_, 1, 1)
            self.cv2 = PaddleConv(c1, c_, 1, 1)
            self.cv3 = PaddleConv(2 * c_, c2, 1)  # act=FReLU(c2)
            self.m = paddle.nn.Sequential(
                *[PaddleBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
            # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

        def forward(self, x):
            return self.cv3(paddle.concat((self.m(self.cv1(x)), self.cv2(x)), axis=1))

    model = PaddleC3(c1=3, c2=3, n=1)
    return model

if __name__ == '__main__':

    tm = get_torch_mm()
    pm = get_paddle_mm()

    reset_parameters(pm, tm)

