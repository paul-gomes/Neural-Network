import math

class ActivationFunction:
    def __init__(self, types='Sigmoid'):
        self.func = self.sigmoid
        self.dfunc = self.dsigmoid

        if types == 'Sigmoid':
            self.func = self.sigmoid
            self.dfunc = self.dsigmoid

    def run(self, x):
        return self.func(x)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # derivative of the sigmoid function, in terms of the output (i.e. y)
    def dsigmoid(self, y):
        return y * (1 - y)


if __name__ == '__main__':
    myfunc = ActivationFunction('Sigmoid')