"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

class Network(minitorch.Module):
    """A simple neural network module consisting of three linear layers and ReLU activations.

    Args:
        hidden_layers (int): The number of neurons in the hidden layers.

    Attributes:
        layer1 (Linear): The first linear layer with input size 2 and output size equal to hidden_layers.
        layer2 (Linear): The second linear layer with input and output size equal to hidden_layers.
        layer3 (Linear): The final linear layer with input size hidden_layers and output size 1.

    Methods:
        forward(x: minitorch.Tensor) -> minitorch.Tensor:
            Defines the forward pass of the network. It applies ReLU activation
            after the first and second layers and a sigmoid activation at the end.
    """
    def __init__(self, hidden_layers: int):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        hidden1 = self.layer1.forward(x).relu()
        hidden2 = self.layer2.forward(hidden1).relu()
        y = self.layer3.forward(hidden2)
        return y.sigmoid()


class Linear(minitorch.Module):
    """A fully connected linear layer that performs a linear transformation on the input data.

    Args:
        in_size (int): The size of each input sample.
        out_size (int): The size of each output sample.

    Attributes:
        weights (RParam): The learnable weights of the layer, initialized with shape (in_size, out_size).
        bias (RParam): The learnable bias, initialized with shape (out_size,).

    Methods:
        forward(x: minitorch.Tensor) -> minitorch.Tensor:
            Applies the linear transformation to the input tensor. The output is
            calculated as a matrix multiplication between the input and the weights,
            followed by adding the bias.
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor):
        (batch_size, in_size) = x.shape

        tmp = (x.view(batch_size, in_size, 1) * self.weights.value).sum(1)
        return tmp.view(batch_size, self.out_size) + self.bias.value


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
