import dynet as dy
import numpy as np


class GumbelSoftmaxTreeLSTM:
    def __init__(self, D_h, D_x, D_c):
        pc = dy.ParameterCollection()
        self.__pc = pc
        self.__D_h = D_h
        self.__D_x = D_x
        self.__D_c = D_c

        self.__W_comp = pc.add_parameters((5 * D_h, 2 * D_h))
        self.__b_comp = pc.add_parameters(5 * D_h)  # todo make sure it's 5 and not 2 like in the paper

        self.__W_leaf = pc.add_parameters((2 * D_h, D_x))
        self.__b_leaf = pc.add_parameters(2 * D_h)
        pass

    def __represent_parent(self, left, right):
        h_l, c_l = left
        h_r, c_r = right
        W, b = dy.parameter(self.__W_comp), dy.parameter(self.__b_comp)
        d = self.__D_h
        hs = dy.concatenate([h_l, h_r])  # 2 * D_h by 1 vector
        temp = W * hs + b

        # Generating i, f_l, f_r, o & g
        i = dy.logistic(temp[0:d])  # sigmoid
        f_l = dy.logistic(temp[d:2*d])
        f_r = dy.logistic(temp[2*d:3*d])
        o = dy.logistic(temp[3*d:4*d])
        g = dy.tanh(temp[4*d:5*d])

        # computing parent data
        c_p = dy.cmult(f_l, c_l) + dy.cmult(f_r, c_r) + dy.cmult(i, g)
        h_p = dy.cmult(o, dy.tanh(c_p))
        return h_p, c_p

    def __parents_of_layer(self, layer):
        return [self.__represent_parent(layer[i], layer[i + 1]) for i in range(len(layer) - 1)]

    def __call__(self, inputs):
        # todo inputs to vectors sized D_x
        inputs = inputs
        D_h = self.__D_h

        # make the first layer: turn each word vector (sized D_x) to a 2 D_h vectors (h, c)
        W_leaf, b_leaf = dy.parameter(self.__W_leaf), dy.parameter(self.__b_leaf)
        layer = [W_leaf * x + b_leaf for x in inputs]
        layer = [(hc[0:D_h], hc[D_h:2*D_h]) for hc in layer]

        while len(layer) > 1:
            pass
        pass
