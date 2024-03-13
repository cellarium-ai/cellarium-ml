import torch

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class GradientReverseModule(torch.nn.Module):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return GradientReverseLayer.apply(input_, self._alpha)


class AdvNet(torch.nn.Module):
    def __init__(self, in_feature=20, hidden_size=20, out_dim=2):
        super(AdvNet, self).__init__()
        self.ad_layer1 = torch.nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = torch.nn.Linear(hidden_size, out_dim)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm1d(hidden_size)
        self.norm2 = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(init_weights)
        self.grl = GradientReverseModule()

    def forward(self, x, reverse=True, if_activation=True):
        if reverse:
            x = self.grl(x)
        x = self.ad_layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        if if_activation:
            y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1