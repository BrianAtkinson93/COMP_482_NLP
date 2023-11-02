import torch as t


def main():
    # demonstrating calculation of derivative of vector function Q
    a = t.tensor([2., 3.], requires_grad=True)
    b = t.tensor([4., 5.], requires_grad=True)

    Q = 2 * a ** 4 - b ** 3

    external_grad = t.tensor([.01, .01])
    Q.backward(gradient=external_grad)

    print(a.grad)
    print(8 * a ** 3 * external_grad[0])

    print(b.grad)
    print(-3 * b ** 2 * external_grad[1])


if __name__ == "__main__":
    main()
