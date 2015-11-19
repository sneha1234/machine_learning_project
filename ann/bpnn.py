
# Adapted from the python implementation by Neil Schemenauer on wikipedia
# and slightly modification.

#!/usr/bin/env python
import math
import random
import numpy as np

random.seed(0)
class Ann:
    def __init__(self, training_data, mu, n_in, n_out, n_hidden):
        self.n_out = n_out
        self.n_in = n_in
        self.n_hidden = n_hidden

        # activations for nodes
        self.Xi = [1.0]*self.n_in
        self.hidden_out = [1.0]*self.n_hidden
        self.final_out = [1.0]*self.n_out

        self.in_weights = np.matrix[[0.0] * self.n_hidden] * self.n_in)
        self.out_weights = np.matrix([[0.0] * self.n_out] * self.n_hidden)
        self.in_current = np.matrix[[0.0] * self.n_hidden] * self.n_in)
        self.out_current = np.matrix([[0.0] * self.n_out] * self.n_hidden)

        for i in range(n_in):
            for j in range(n_hidden):
                self.in_weights[i,j] = rand(-0.05, 0.05)
        for i in range(n_hidden):
            for j in range(n_out):
                self.out_weights[i,j] = rand(-0.5, 0.5)

    def feed_forward(self, inp):
        # feed inputs through input nodes
        for i in range(self.n_in - 1): # last input is 1
            self.Xi[i] = inp[i]

        # input from in nodes to hidden nodes is weight and activated
        for j in range(self.n_hidden):
            sum = 0.0
            for i in range(self.n_in):
                sum = sum + self.Xi[i] * self.in_weights[i,j]
            self.hidden_out[j] = math.tanh(sum)

        # input from hidden nodes to output nodes is further weight and activated
        for k in range(self.n_out):
            sum = 0.0
            for j in range(self.n_hidden):
                sum = sum + self.hidden_out[j] * self.out_weights[j, k]
            self.final_out[k] = math.tanh(sum)

    def backPropagate(self, targets, N, M):
        if len(targets) != self.n_out:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.n_out
        for k in range(self.n_out):
            error = targets[k]-self.final_out[k]
            output_deltas[k] = ( 1.0 - self.final_out[k]**2) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            error = 0.0
            for k in range(self.n_out):
                error = error + output_deltas[k]*self.out_weights[j][k]
            hidden_deltas[j] = dsigmoid(self.hidden_out[j]) * error

        # update output weights
        for j in range(self.n_hidden):
            for k in range(self.n_out):
                change = output_deltas[k]*self.hidden_out[j]
                self.out_weights[j][k] = self.out_weights[j][k] + N*change + M*self.out_current[j][k]
                self.out_current[j][k] = change
                #print N*change, M*self.out_current[j][k]

        # update input weights
        for i in range(self.n_in):
            for j in range(self.n_hidden):
                change = hidden_deltas[j]*self.Xi[i]
                self.in_weights[i][j] = self.in_weights[i][j] + N*change + M*self.in_current[i][j]
                self.in_current[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.final_out[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.n_in):
            print(self.in_weights[i])
        print()
        print('Output weights:')
        for j in range(self.n_hidden):
            print(self.out_weights[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)



if __name__ == '__main__':
    demo()
