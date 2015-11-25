#!/usr/bin/env python

# Adapted from the python implementation by Neil Schemenauer on wikipedia
# and slightly modified.

import math
import random
import numpy as np
import sys
import pandas as pd

random.seed(0)
class Ann:
    def __init__(self, training_data, mu, mo, n_in, n_out, n_hidden):
        self.training_data = training_data
        self.mu = mu # learning rate
        self.mo = mo #momentum
        self.n_out = n_out
        self.n_in = n_in + 1 # plus one bias node
        self.n_hidden = n_hidden

        # activations for nodes
        self.Xi = [1.0]*self.n_in
        self.hidden_out = [1.0]*self.n_hidden
        self.final_out = [1.0]*self.n_out

        self.in_weights = np.matrix([[0.0] * self.n_hidden] * self.n_in)
        self.out_weights = np.matrix([[0.0] * self.n_out] * self.n_hidden)
        self.in_current = np.matrix([[0.0] * self.n_hidden] * self.n_in)
        self.out_current = np.matrix([[0.0] * self.n_out] * self.n_hidden)

        for i in range(self.n_in):
            for j in range(n_hidden):
                self.in_weights[i,j] = random.uniform(-0.05, 0.05)
        for i in range(self.n_hidden):
            for j in range(self.n_out):
                self.out_weights[i,j] = random.uniform(-0.5, 0.5)
        #print self.in_weights
        #print self.out_weights[0,0], self.out_weights[1,0]
    def feed_forward(self, inp):
        if len(inp) != self.n_in-1:
            raise ValueError('wrong number of inputs')

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

        return self.final_out[:]

    def backPropagate(self, targets):
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
                error = error + output_deltas[k]*self.out_weights[j,k]
            hidden_deltas[j] = (1.0 - self.hidden_out[j]**2) * error

        # update output weights
        for j in range(self.n_hidden):
            for k in range(self.n_out):
                change = output_deltas[k]*self.hidden_out[j]
                self.out_weights[j,k] = self.out_weights[j,k] + self.mu * change + self.mo * self.out_current[j,k]
                self.out_current[j,k] = change
                #print N*change, M*self.out_current[j][k]

        # update input weights
        for i in range(self.n_in):
            for j in range(self.n_hidden):
                change = hidden_deltas[j]*self.Xi[i]
                self.in_weights[i,j] = self.in_weights[i,j] + self.mu *change + self.mo * self.in_current[i,j]
                self.in_current[i,j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.final_out[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.feed_forward(p[0]))

    def weights(self):
        print('Input weights:')
        print self.in_weights
        print ""
        print('Output weights:')
        print self.out_weights

    def train(self, iterations=2000):
        # N: learning rate
        # M: momentum factor

        for i in range(iterations):
            error = 0.0
            for p in self.training_data:
                inputs = p[0]
                targets = p[1]
                self.feed_forward(inputs)
                error = error + self.backPropagate(targets)
            #if i % 100 == 0:
                #print('error %-.5f' % error)

def demo(data):
    mu=0.5 #learning rate
    mo=0.1 # moment

    # Teach network XOR function
    pat = [ # -- yes | no
        [[0, 0, 1], [1, 0]],
        [[0, 1, 0], [1, 0]],
        [[1, 1, 1], [0, 1]],
        [[1, 0, 0], [1, 0]],
        [[1, 1, 0], [1, 0]],
        [[0, 1, 1], [1, 0]],
        [[1, 0, 1], [1, 0]]

    ]

    # create a network with 4 input, 3 output and 4 hidden nodes
    n = Ann(data, mu, mo, 4, 3, 4)
    # train it with some patterns
    n.train()
    n.weights()

    # test it
    n.test([[[5.9,3.0,5.1,1.8]]])

if __name__ == '__main__':
	pd.set_option("display.precision", 2)
	df = pd.read_csv(sys.argv[1], quoting=1, header=None)
	train_data = []
	for i, row in df.iterrows():
		cls = [0,0,0]
		cls[int(row[4])] = 1
		train_data.append([[row[0], row[1],row[2],row[3]], cls])
		# print train_data
	demo(train_data)
