# Example to use a user-defined Module as a hedging model
# Here we show an example of No-Transaction Band Network,
# which is proposed in Imaki et al. 21.

import sys
import logging
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Module

sys.path.append("..")
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import Clamp
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron

# from pfhedge.nn import NoTransactionBandNet

# Add logging configuration
start_time = time.time()
logging.basicConfig(
    format='[%(elapsed).3f s] %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)

# Create a custom logging filter to add elapsed time
class ElapsedTimeFilter(logging.Filter):
    def filter(self, record):
        record.elapsed = time.time() - start_time
        return True

# Add the filter to the root logger
logging.getLogger().addFilter(ElapsedTimeFilter())

class NoTransactionBandNet(Module):
    """Initialize a no-transaction band network.

    The `forward` method returns the next hedge ratio.

    Args:
        derivative (pfhedge.instruments.BaseDerivative): The derivative to hedge.

    Shape:
        - Input: :math:`(N, H_{\\text{in}})`, where :math:`(N, H_{\\text{in}})` is the
        number of input features. See `inputs()` for the names of input features.
        - Output: :math:`(N, 1)`.

    Examples:

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption

        >>> derivative = EuropeanOption(BrownianStock(cost=1e-4))
        >>> m = NoTransactionBandNet(derivative)
        >>> m.inputs()
        ['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge']
        >>> input = torch.tensor([
        ...     [-0.05, 0.1, 0.2, 0.5],
        ...     [-0.01, 0.1, 0.2, 0.5],
        ...     [ 0.00, 0.1, 0.2, 0.5],
        ...     [ 0.01, 0.1, 0.2, 0.5],
        ...     [ 0.05, 0.1, 0.2, 0.5]])
        >>> m(input)
        tensor([[0.2232],
                [0.4489],
                [0.5000],
                [0.5111],
                [0.7310]], grad_fn=<SWhereBackward>)
    """

    def __init__(self, derivative):
        super().__init__()

        self.delta = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()
        self.counter = 0.

    def inputs(self):
        return self.delta.inputs() + ["prev_hedge"]

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.delta(input[..., :-1])
        width = self.mlp(input[..., :-1])

        min = delta - fn.leaky_relu(width[..., [0]])
        max = delta + fn.leaky_relu(width[..., [1]])

        # print(f"made {self.counter}  step forward")
        self.counter += 1

        return self.clamp(prev_hedge, min=min, max=max)


if __name__ == "__main__":

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")
    
    torch.manual_seed(42)

    # Prepare a derivative to hedge
    derivative = EuropeanOption(BrownianStock(cost=5.5e-4, sigma=0.22)).to(DEVICE)

    # Create your hedger
    model = NoTransactionBandNet(derivative).to(DEVICE)
    hedger = Hedger(model, model.inputs()).to(DEVICE)

    # Fit and price
    hedger.fit(derivative, n_paths=10000, n_epochs=200)
    price = hedger.price(derivative, n_paths=10000)
    logging.info(f"Price={price:.5e}")

    
    # Generate data for plotting
    log_moneyness = torch.linspace(-0.5, 0.5, steps=100)
    bs_delta = []
    ntb_min = []
    ntb_max = []

    for lm in log_moneyness:
        input_features = torch.tensor([[lm.item(), 15 / 365, 0.22, 0.0]])
        bs_delta.append(model.delta(input_features[:, :-1]).item())
        ntb_band = model(input_features).detach()
        ntb_min.append(ntb_band[:, 0].item())
        ntb_max.append(ntb_band[:, 1].item())

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(log_moneyness, bs_delta, label="Black-Scholes delta", linestyle="dashed")
    plt.plot(log_moneyness, ntb_min, label="No-transaction band (min)", linestyle="solid")
    plt.plot(log_moneyness, ntb_max, label="No-transaction band (max)", linestyle="solid")

    plt.xlabel("Log moneyness")
    plt.ylabel("Hedging strategy / Band")
    plt.title("No-transaction-band for European Option")
    plt.legend()
    plt.grid(True)
    plt.show()
    logging.info("Done")
    