import os
import math
from typing import Tuple, Any

import torch as t
import pandas as pd
import matplotlib.pyplot as plt
import random

from pandas import DataFrame
from torch import Tensor


def check_not_inf_nan(nums) -> bool:
    """
    Check if any value in the list is either infinity or NaN.

    Args:
        nums (list[float|int]): List of numbers to check.

    Returns:
        bool: True if any number in the list is either infinity or NaN, False otherwise.
    """
    for v in nums:
        if math.isnan(v) or math.isinf(v):
            return True
    return False


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


def data_setup() -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, DataFrame]:
    """
    Set up the data required for training.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, DataFrame]: Tensors representing x1, x2, y1, y2, params, and the dataframe.
    """
    # --- data setup ---
    data_path = os.path.abspath('data/points.csv')
    df = pd.read_csv(data_path)

    m1 = random.uniform(-1, 1)
    b1 = random.uniform(0, 10)
    m2 = random.uniform(-1, 1)
    b2 = random.uniform(0, 10)
    params = t.tensor([m1, b1, m2, b2], requires_grad=True)

    x1 = t.tensor(df.iloc[0:10, 1].values)
    x2 = t.tensor(df.iloc[10:20, 1].values)
    y1 = t.tensor(df.iloc[0:10, 2].values)
    y2 = t.tensor(df.iloc[10:20, 2].values)

    x = t.cat((x1, x2), 0)

    return x1, x2, y1, y2, params, df


def training(x1, x2, y1, y2, params) -> Tuple[Any, Any]:
    """
    Train the model using the provided data.

    Args:
        x1, x2 (Tensor): Input data.
        y1, y2 (Tensor): Target data.
        params (Tensor): Initial parameters for training.

    Returns:
        Tuple[Tensor, Tensor]: Tensors representing l1 and l2 after training.
    """
    # --- training setup ---

    steps = 10000

    # --- training ---
    l1 = None
    l2 = None

    for step in range(steps):
        learning_rate = 2.0 / steps
        l1 = params[0] * x1 + params[1]
        l2 = params[2] * x2 + params[3]
        objective = t.sum(t.square(y1 - l1)) + t.sum(t.square(y2 - l2))
        external_grad = t.tensor(learning_rate)
        objective.backward(gradient=external_grad)
        print(
            'step {j:5d}: loss {v:.10f} --- w [{w1:.10f}, {w2:.10f}] --- b [{b1:.10f}, {b2:.10f}] '
            .format(j=step, v=objective.item(), w1=params[0], w2=params[2], b1=params[1], b2=params[3])
        )
        if check_not_inf_nan(params.tolist()):
            print('Exiting because grad became unstable')
            break
        with t.no_grad():
            params -= params.grad
            params.grad = None  # do not accumulate gradients during training

    return l1, l2


def display_results(x1, x2, y1, y2, l1, l2, df) -> None:
    """
    Display the results of the training.

    Args:
        x1, x2, y1, y2, l1, l2 (Tensor): Tensors representing data and training results.
        df (DataFrame): Dataframe with the data points.

    Returns:
        None
    """
    # --- display results ---

    t.set_grad_enabled(False)  # turn off gradient processing after training

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

    plt.show()

    plt.close("all")


def main():
    """Main Loop"""
    # Set up the data
    x1, x2, y1, y2, params, df = data_setup()

    # Train the model
    l1, l2 = training(x1, x2, y1, y2, params)

    # Display the results
    display_results(x1, x2, y1, y2, l1, l2, df)


if __name__ == "__main__":
    main()
