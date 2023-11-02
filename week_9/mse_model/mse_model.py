# import math
# import random
import sys
import os
import logging

import torch as t
import pandas as pd
import matplotlib.pyplot as plt

from torch.autograd import Variable


class LinearRegression(t.nn.Module):
    """
    Custom Linear Regression model using PyTorch.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialize the linear regression model.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.

        Returns:
            None
        """
        super(LinearRegression, self).__init__()
        self.linear = t.nn.Linear(input_size, output_size)
        with t.no_grad():
            self.linear.bias.uniform_(-20, 20)

    def get_bias(self) -> t.Tensor:
        """
        Retrieve the bias of the model.

        Returns:
            Tensor: The bias tensor.
        """
        return self.linear.bias.to(device='cpu')

    def get_weights(self) -> t.Tensor:
        """
        Retrieve the weights of the model.

        Returns:
            Tensor: The weights' tensor.
        """
        return self.linear.weight.to(device='cpu')

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.linear(x)
        return out

    def train(self, x: t.Tensor, y: t.Tensor, steps: int = 10000) -> None:
        """
        Train the model using the provided data.

        Args:
            x (Tensor): Input tensor.
            y (Tensor): Target tensor.
            steps (int): Number of training steps.

        Returns:
            None
        """
        log.info('Starting training...')
        # start training
        learning_rate = 30.0 / steps
        t.set_grad_enabled(True)
        if t.cuda.is_available():  # use GPU, if possible
            self.cuda()
        # loss function is minimum sum of square error differences (minimize while training)
        criterion = t.nn.MSELoss()
        # stochastic gradient descent
        optimizer = t.optim.SGD(self.parameters(), lr=learning_rate)
        if t.cuda.is_available():
            inputs = Variable(x.reshape(-1, 1).cuda())  # use GPU, if possible
            labels = Variable(y.reshape(-1, 1).cuda())
        else:
            inputs = Variable(x.reshape(-1, 1))  # otherwise, use CPU
            labels = Variable(y.reshape(-1, 1))
        for step in range(steps):
            learning_rate = 30.0 / steps
            optimizer.zero_grad()  # do not accumulate gradients across steps
            outputs = self(inputs.float())  # approximated values
            w = self.get_weights().squeeze()
            b = self.get_bias()
            leg = w * x + b
            loss = criterion(outputs, labels)
            # print(t.sum(t.square(leg - Y))) # see the MSE if you need, which should be same as loss
            loss.backward()  # calculate gradient
            optimizer.step()  # move weights and bias opposite direction from gradient
            log.debug(
                'step {j:5d}: loss {v:.10f} --- mse {mse:.10f} --- w {wght:.10f} --- b {bias:.10f}'
                .format(j=step, v=loss.item(), mse=t.sum(t.square(leg - y)), wght=w, bias=b.item())
            )
        # done with training
        t.set_grad_enabled(False)


def plot_vertical(x, y, line, scatterplot) -> None:
    """
    Plot a vertical line on the scatterplot using the provided x and y values.

    Args:
        x (Tensor): X-coordinates.
        y (Tensor): Y-coordinates.
        line (list[DataFrame]): List to append the line data.
        scatterplot (Any): The scatterplot to which the line will be added.

    Returns:
        None
    """
    line.append(pd.DataFrame({'X': x, 'dY': y}))
    line[-1].plot(x='X', y='dY', color='yellow', ax=scatterplot, legend=False)


def main(logger) -> int:
    """
    Main function to execute the program.
    Loads data, trains models, and plots the results.

    Returns:
        None
    """
    log.info("Main Func...")

    fp = 'data/points.csv'
    logger.info(f"Loading {fp}...")
    data_path = os.path.abspath(fp)
    df = pd.read_csv(data_path)

    x1 = t.tensor(df.iloc[0:10, 1].values).float()
    x2 = t.tensor(df.iloc[10:20, 1].values).float()
    y1 = t.tensor(df.iloc[0:10, 2].values).float()
    y2 = t.tensor(df.iloc[10:20, 2].values).float()

    input_dim = 1
    output_dim = 1

    model1 = LinearRegression(input_dim, output_dim)
    model1.train(x1, y1)

    model2 = LinearRegression(input_dim, output_dim)
    model2.train(x2, y2)

    w1 = model1.get_weights().squeeze()
    w2 = model2.get_weights().squeeze()
    l1 = w1 * x1 + model1.get_bias()
    l2 = w2 * x2 + model2.get_bias()

    scatterplot = df.plot.scatter(x='a', y='b')
    df_line1 = pd.DataFrame({'Lx1': x1, 'Ly1': l1})
    df_line2 = pd.DataFrame({'Lx2': x2, 'Ly2': l2})

    df_lines1 = []
    df_lines2 = []
    for i in range(len(x1)):
        plot_vertical(
            x1[i:i + 1].repeat(2),  # x
            t.cat((y1[i:i + 1], l1[i:i + 1]), 0),  # y
            df_lines1, scatterplot
        )
        plot_vertical(
            x2[i:i + 1].repeat(2),
            t.cat((y2[i:i + 1], l2[i:i + 1]), 0),
            df_lines2, scatterplot
        )

    df_line1.plot(x='Lx1', y='Ly1', kind='line', color='purple', ax=scatterplot)
    df_line1.plot.scatter(x='Lx1', y='Ly1', color='purple', ax=scatterplot)
    df_line2.plot(x='Lx2', y='Ly2', kind='line', color='green', ax=scatterplot)
    df_line2.plot.scatter(x='Lx2', y='Ly2', color='green', ax=scatterplot)

    logger.info('Showing Plot...')
    plt.show()

    plt.close("all")

    return 0


def setup_logger(log_level):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.NOTSET, handlers=[])

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'{__name__}.log')
    c_handler.setLevel(log_level)
    f_handler.setLevel(log_level)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


if __name__ == "__main__":
    log = setup_logger(log_level=logging.INFO)
    log.info("Starting run...")
    sys.exit(main(log))
