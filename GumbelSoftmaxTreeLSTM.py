import dynet as dy
import numpy as np
from itertools import izip


class SimpleGumbelSoftmaxTreeLSTM:
    def __init__(self, D_h, D_x, temperatue=1.0, use_leaf_lstm=False, lstm_layers=1, use_bilstm=False):
        """
        Ctor.
        :param D_h: Size of inner representation vector.
        :param D_x: Size of input vectors.
        :param temperatue: The temperature for the gumbel softmax trick.
        :param use_leaf_lstm: If true uses an LSTM RNN to encode the leaf nodes.
        :param lstm_layers: How many layers the lstm will use, if used.
        :param use_bilstm: If true and use_leaf_lstm is true, uses a bilstm instead of a one way lstm.
        (This doubles D_h!!)
        """
        pc = dy.ParameterCollection()
        self.pc = pc
        self.__D_h = D_h
        self.__D_x = D_x
        self.__use_leaf_lstm = use_leaf_lstm
        self.__use_bilstm = use_leaf_lstm and use_bilstm

        self.__W_comp = pc.add_parameters((5 * D_h, 2 * D_h))
        self.__b_comp = pc.add_parameters(5 * D_h)  # papar is worng, should be 5*D_h and not 2*D_h

        # leaf "encoding"
        if not use_leaf_lstm:  # uses a linear layer
            self.__W_leaf = pc.add_parameters((2 * D_h, D_x))
            self.__b_leaf = pc.add_parameters(2 * D_h)
        else:
            self.__leaf_lstm = dy.LSTMBuilder(lstm_layers, D_x, D_h, pc)
            if self.__use_bilstm:
                self.__bw_leaf_lstm = dy.LSTMBuilder(lstm_layers, D_x, D_h, pc)

        self.__query_vec = pc.add_parameters((D_h, 1))
        self.__temperatue = temperatue
        pass

    def __represent_parent(self, left, right):
        """
        Represent the parent of 2 nodes.
        :param left: Tuple - (h_l, c_l)
        :param right: Tuple - (h_r, c_r)
        :return: Tuple - (h_p, c_p) - representation of the parents.
        """
        h_l, c_l = left
        h_r, c_r = right
        W, b = dy.parameter(self.__W_comp), dy.parameter(self.__b_comp)
        d = self.__D_h
        hs = dy.concatenate([h_l, h_r])  # 2 * D_h by 1 vector
        temp = W * hs + b

        # Generating i, f_l, f_r, o & g
        i = dy.logistic(temp[0:d])  # sigmoid
        f_l = dy.logistic(temp[d:2 * d])
        f_r = dy.logistic(temp[2 * d:3 * d])
        o = dy.logistic(temp[3 * d:4 * d])
        g = dy.tanh(temp[4 * d:5 * d])

        # computing parent data
        c_p = dy.cmult(f_l, c_l) + dy.cmult(f_r, c_r) + dy.cmult(i, g)
        h_p = dy.cmult(o, dy.tanh(c_p))
        return h_p, c_p

    def __parents_of_layer(self, layer):
        """
        Represent all the possible parents in layer.
        :param layer: List holding tuples of (h, c) of nodes.
        :return: A list of tuples (h_p, c_p). The i-th tuple is the representation of the parent of the i-th and
        (i+1)-th nodes in layer.
        """
        return [self.__represent_parent(layer[i], layer[i + 1]) for i in xrange(len(layer) - 1)]

    @staticmethod
    def gumbel_softmax(pis, temperatue=1.0):
        """
        Computes a gumbel softmax on probabilities.
        :param pis: Dynet vector of probabilities to chose from.
        :param temperatue: Temperature of the softmax.
        :return: Pis gumbel-sofrmaxed (Dynet vector)
        """
        u = dy.random_uniform(pis.dim()[0], 0.0, 1.0)
        g = -dy.log(-dy.log(u))
        y = dy.exp((dy.log(pis) + g) / temperatue)
        y = dy.cdiv(y, dy.sum_elems(y))
        return y

    def __call__(self, inputs, test=False, renew_cg=True):
        """
        Represent the sentence using an lstm tree by choosing at each layer a couple of nodes to unite using gumbel
        softmax.
        :param inputs: List of dynet vectors (words in sentence).
        :param test: If true doesn't uses the gumbel softmax but just chosses the max score.
        :param renew_cg: If false doesn't renew the computation graph
        :return: The (h, c) representation of the sentence.
        """
        if renew_cg:
            dy.renew_cg()
        D_h = self.__D_h

        # make the first layer: turn each word vector (sized D_x) to a 2 D_h vectors (h, c)
        if not self.__use_leaf_lstm:
            W_leaf, b_leaf = dy.parameter(self.__W_leaf), dy.parameter(self.__b_leaf)
            layer = [W_leaf * x + b_leaf for x in inputs]
            layer = [(hc[0:D_h], hc[D_h:2 * D_h]) for hc in layer]
        else:
            s0 = self.__leaf_lstm.initial_state()
            if self.__use_bilstm:
                bw_s0 = self.__bw_leaf_lstm.initial_state()

            h0 = dy.zeros(self.__D_x)
            c0 = dy.zeros(self.__D_x)
            layer = [(h0, c0)]
            last_h, last_c = h0, c0
            if self.__use_bilstm:
                last_bw_h, last_bw_c = h0, c0

            length = len(inputs)
            for i in xrange(1, length):
                lstm_input = [inputs[i], last_h, last_c]
                hc = s0.transduce(lstm_input)
                h, c = hc[0:D_h], hc[D_h:2 * D_h]
                last_h, last_c = h, c

                if self.__use_bilstm:
                    bw_lstm_input = [inputs[length - 1 - i], last_bw_h, last_bw_c]
                    bw_hc = bw_s0.transduce(bw_lstm_input)
                    bw_h, bw_c = bw_hc[0:D_h], bw_hc[D_h:2 * D_h]
                    h, c = dy.concatenate(h, bw_h), dy.concatenate(c, bw_c)
                    last_bw_h, last_bw_c = bw_h, bw_c

                layer.append((h, c))

        q = dy.parameter(self.__query_vec)  # query vector
        while len(layer) > 1:
            parents = self.__parents_of_layer(layer)  # all possible parents of pairs in layer

            # creating v_1,...,v_M_t+1, Eq. (12) in the paper
            parents_scores = map(lambda (h, c): dy.dot_product(h, q), parents)
            score_sum = dy.esum(parents_scores)
            parents_scores = dy.concatenate(parents_scores)
            parents_scores = dy.cdiv(parents_scores, score_sum)

            if not test:
                y = self.gumbel_softmax(parents_scores, self.__temperatue)
            else:
                y = parents_scores

            y = y.npvalue()
            # y_st = np.zeros(y.shape)
            # y_st[y.argmax()] = 1.0
            y = y.argmax()
            layer = layer[:y] + [parents[y]] + layer[y + 2:]

        return layer[0]


class SimpleSNLIGumbelSoftmaxTreeLSTM:
    def __init__(self, D_h, D_x, D_c, mlp_hidden_layer_size, mlp_hidden_layers=1, temperatue=1.0, use_leaf_lstm=False,
                 lstm_layers=1, use_bilstm=False):
        """
        Ctor.
        :param D_h: Size of inner representation vector.
        :param D_x: Size of input vectors.
        :param temperatue: The temperature for the gumbel softmax trick.
        :param use_leaf_lstm: If true uses an LSTM RNN to encode the leaf nodes.
        :param lstm_layers: How many layers the lstm will use, if used.
        :param use_bilstm: If true and use_leaf_lstm is true, uses a bilstm instead of a one way lstm.
        (This doubles D_h!!)
        :param D_c: Size of MLP layers
        :param mlp_hidden_layer_size: likewise
        :param mlp_hidden_layers: Number of hidden layers of MLP
        """
        self.__treeLSTM = SimpleGumbelSoftmaxTreeLSTM(D_h, D_x, temperatue, use_leaf_lstm, lstm_layers, use_bilstm)
        self.__D_c = D_c
        pc = self.__treeLSTM.pc
        self.ENTAILMENT, self.CONTRADICTION, self.NEUTRAL = 0, 1, 2
        self.__W_cl_f = [pc.add_parameters((1, D_c)), pc.add_parameters((1, D_c)), pc.add_parameters((1, D_c))]
        self.__b_cl_f = [pc.add_parameters(1), pc.add_parameters(1), pc.add_parameters(1)]

        # creates an mlp with n (n = {mlp_hidden_layers}) hidden layers and an input layer
        # input layer:
        self.__mlp = [(pc.add_parameters((mlp_hidden_layer_size, 4 * D_h)), pc.add_parameters(mlp_hidden_layer_size))]
        # n - 1 hidden layers:
        self.__mlp += [(pc.add_parameters((mlp_hidden_layer_size, mlp_hidden_layer_size)),
                        pc.add_parameters(mlp_hidden_layer_size)) for _ in xrange(mlp_hidden_layers - 1)]
        # last hidden layer to output:
        self.__mlp += [(pc.add_parameters((D_c, mlp_hidden_layer_size)), pc.add_parameters(mlp_hidden_layer_size))]
        self.__mlp_activation = dy.rectify  # ReLu

    def get_parameter_collection(self):
        """
        :return: The model's dynet.ParameterCollection
        """
        return self.__treeLSTM.pc

    def __apply_mlp(self, f):
        """
        Applying the MLP to the input f, using the activation function (default is ReLu).
        :param f: Input to MLP
        :return: The MLP's output on f.
        """
        g = self.__mlp_activation
        x = f
        for pW, pb in self.__mlp[:-1]:
            W, b = dy.parameter(pW), dy.parameter(pb)
            x = g(W * x + b)
        pW, pb = self.__mlp[-1]
        W, b = dy.parameter(pW), dy.parameter(pb)
        return W * x + b

    def __call__(self, premise, hypothesis, test=False, use_dropout=False, dropout_prob=0.1):
        """
        Using the tree lstm represents the 2 sentences, then takes those represenation after processing and puts them
        into a MLP, then calculates probabilities for their relationship.
        :param premise: Premise sentence (List of numpy vectors)
        :param hypothesis: Hypothesis sentence (List of numpy vectors)
        :param test: If true doesn't uses the gumbel softmax but just chosses the max score.
        :param use_dropout: If true uses dropout on the input vectors, the MLP input and the output.
        :param dropout_prob: The probability for the dropout, if used.
        :return: Vector of probabilities for each class.
        """
        dy.renew_cg()

        premise = [dy.inputTensor(v) for v in premise]
        hypothesis = [dy.inputTensor(v) for v in hypothesis]
        if use_dropout:  # dropout sentences
            premise = [dy.dropout(v, dropout_prob) for v in premise]
            hypothesis = [dy.dropout(v, dropout_prob) for v in hypothesis]

        h_pre, c_pre = self.__treeLSTM(premise, renew_cg=False, test=test)
        h_hyp, c_hyp = self.__treeLSTM(hypothesis, renew_cg=False, test=test)

        f = dy.concatenate([h_pre, h_hyp, dy.abs(h_pre - h_pre), dy.cmult(h_pre, h_hyp)])
        if use_dropout:
            f = dy.dropout(f, dropout_prob)
        a = self.__apply_mlp(f)
        if use_dropout:
            a = dy.dropout(a, dropout_prob)

        probs = [0, 0, 0]
        for i in [self.ENTAILMENT, self.CONTRADICTION, self.NEUTRAL]:
            W, b = dy.parameter(self.__W_cl_f[i]), dy.parameter(self.__b_cl_f[i])
            probs[i] = W * a + b
        probs = dy.concatenate(probs)
        return dy.softmax(probs)

    def loss_on(self, premise, hypothesis, expected, use_dropout=False, dropout_prob=0.1):
        """
        Computes loss on the sentences.
        :param premise: Premise sentence (List of numpy vectors)
        :param hypothesis: Hypothesis sentence (List of numpy vectors)
        :param expected: Expected label of those sentences.
        :param use_dropout: If true uses dropout on the input vectors, the MLP input and the output.
        :param dropout_prob: The probability for the dropout, if used.
        :return: The loss.
        """
        probs = self(premise, hypothesis, test=False, use_dropout=use_dropout, dropout_prob=dropout_prob)
        return -dy.log(probs[expected])

    def predict(self, premise, hypothesis):
        """
        Predicts on the sentences.
        :param premise: Premise sentence (List of numpy vectors)
        :param hypothesis: Hypothesis sentence (List of numpy vectors)
        :return: The index of the label.
        """
        return self(premise, hypothesis, test=True).npvalue().argmax()


"######################################################################################################################"
"################################################ BATCHED CLASSES #####################################################"
"######################################################################################################################"


class GumbelSoftmaxTreeLSTM:
    def __init__(self, D_h, D_x, temperatue=1.0, use_leaf_lstm=False, lstm_layers=1, use_bilstm=False):
        """
        Ctor. This is an optimized batched version of the SimpleGumbelSoftmaxTreeLSTM.
        :param D_h: Size of inner representation vector.
        :param D_x: Size of input vectors.
        :param temperatue: The temperature for the gumbel softmax trick.
        :param use_leaf_lstm: If true uses an LSTM RNN to encode the leaf nodes.
        :param lstm_layers: How many layers the lstm will use, if used.
        :param use_bilstm: If true and use_leaf_lstm is true, uses a bilstm instead of a one way lstm.
        (This doubles D_h!!)
        """
        pc = dy.ParameterCollection()
        self.pc = pc
        self.__D_h = D_h
        self.__D_x = D_x
        # self.__batch_size = batch_size
        self.__use_leaf_lstm = use_leaf_lstm
        self.__use_bilstm = use_leaf_lstm and use_bilstm

        # leaf "encoding"
        if not use_leaf_lstm:
            self.__W_leaf = pc.add_parameters((2 * D_h, D_x))
            self.__b_leaf = pc.add_parameters(2 * D_h)
        else:
            self.__leaf_lstm = dy.LSTMBuilder(lstm_layers, D_x, D_h, pc)
            if use_bilstm:
                self.__bw_leaf_lstm = dy.LSTMBuilder(lstm_layers, D_x, D_h, pc)
                D_h = 2 * D_h

        self.__W_comp = pc.add_parameters((5 * D_h, 2 * D_h))
        self.__b_comp = pc.add_parameters(5 * D_h)  # todo make sure it's 5 and not 2 like in the paper

        self.__query_vec = pc.add_parameters((D_h, 1))  #, init=dy.NormalInitializer(mean=0, var=0.01**2))
        self.__temperatue = temperatue
        pass

    def __represent_parent(self, lefts, rights):
        """
        Represents all the possible parents in a sentence sized n, by concatting
        :param lefts: An D_h by n - 1 matrix, which is sentence[:-1]
        :param rights: An D_h by n - 1 matrix, which is sentence[1:]
        :return: hs, cs which are the h, c representation for each couple (ith couple is left[i], right[i]).
        Both hs and cs are D_h by n-1 matrices
        """
        h_l, c_l = lefts
        h_r, c_r = rights
        W, b = dy.parameter(self.__W_comp), dy.parameter(self.__b_comp)
        hs = dy.concatenate([h_l, h_r])  # 2 * D_h by n - 1 matrix
        temp = W * hs + b

        # Generating i, f_l, f_r, o & g - D_h by n - 1 matrices
        d = self.__D_h
        i = dy.logistic(temp[0:d])  # sigmoid
        f_l = dy.logistic(temp[d:2 * d])
        f_r = dy.logistic(temp[2 * d:3 * d])
        o = dy.logistic(temp[3 * d:4 * d])
        g = dy.tanh(temp[4 * d:5 * d])

        # computing parent data
        c_p = dy.cmult(f_l, c_l) + dy.cmult(f_r, c_r) + dy.cmult(i, g)
        h_p = dy.cmult(o, dy.tanh(c_p))
        return [h_p, c_p]

    def __parents_of_layer(self, layer):
        """
        Represents all the possible parents in the layer
        :param layer: Matrix holding all the (h,c) representations of the nodes in sentence.
        :return: A matrix of the (h, c) representations of all the parents.
        """
        d, n = layer.dim()[0]
        hs, cs = dy.select_rows(layer, range(d / 2)), dy.select_rows(layer, range(d / 2, d))
        lefts = (dy.select_cols(hs, range(n - 1)), dy.select_cols(cs, range(n - 1)))
        rights = (dy.select_cols(hs, range(1, n)), dy.select_cols(cs, range(1, n)))
        return dy.concatenate(self.__represent_parent(rights, lefts))  # it was _cols

    @staticmethod
    def gumbel_softmax(pis, temperatue=1.0):
        """
        Computes a gumbel softmax on probabilities.
        :param pis: Dynet vector of probabilities to chose from.
        :param temperatue: Temperature of the softmax.
        :return: vecor sized pis of ys
        """
        u = dy.random_uniform(pis.dim()[0], 0.0, 1.0)
        g = -dy.log(-dy.log(u))
        y_temp = dy.exp((dy.log(pis) + g) / temperatue)
        y = dy.cdiv(y_temp, dy.sum_elems(y_temp))
        return y

    def __y_st_before_argmax(self, parents):
        """
        For each parent computes query vector dot h representation plus gumbel noise.
        :param parents: All the (h,c) parents in a single matrix.
        :return: A matrix of the values for the Gumbel max trick.
        """
        epsilon = 1e-20
        q = dy.parameter(self.__query_vec)  # query vector
        hs = dy.select_rows(parents, range(self.__D_h))
        u = dy.random_uniform((1, hs.dim()[0][1]), 0, 1)
        g = -dy.log(-dy.log(u + epsilon) + epsilon)
        return dy.concatenate_cols([dy.dot_product(dy.select_cols(hs, [i]), q) for i in range(hs.dim()[0][1])]) + g

    def __parents_scores(self, parents):
        """
        Computes scores for the parents.
        :param parents: All the (h,c) parents in a single matrix.
        :return: A vector with each parent's score.
        """
        q = dy.parameter(self.__query_vec)  # query vector
        hs = dy.select_rows(parents, range(self.__D_h))
        return dy.concatenate([dy.exp(dy.dot_product(dy.select_cols(hs, [i]), q)) for i in range(hs.dim()[0][1])])

    @staticmethod
    def cumsum(vec):
        """
        Computes the cum sum vector [c_1,...,c_n] of [v_1,...,v_n] where c_i = v_1 + v_2 + ... + v_i.
        :type vec: dy.Expression
        :param vec: Dynet expression (vector)
        :return: A n sized vector of cumsum(v)
        """
        c = [vec[0]]
        for i in xrange(1, vec.dim()[0][0]):
            c.append(c[i - 1] + vec[i])
        return dy.concatenate(c)  # vector

    def __call__(self, inputs, test=False, renew_cg=True):
        """
        Represents each sentence in batch as (h,c) using tree lstm and gumbel softmax.
        :param inputs: A batch of sentences, each is a list of numpy vectors
        :param test: If true doesn't uses the gumbel softmax but just chosses the max score.
        :param renew_cg: If false doesn't renew the computation graph
        :return: A list of the (h, c) representation for each sentence.
        """
        if renew_cg:
            dy.renew_cg()
        D_h = self.__D_h

        # make the first layer: turn each word vector (sized D_x) to a 2 D_h vectors (h, c)
        if not self.__use_leaf_lstm:
            W_leaf, b_leaf = dy.parameter(self.__W_leaf), dy.parameter(self.__b_leaf)
            layer = [[W_leaf * x + b_leaf for x in inp] for inp in inputs]
            layer = [dy.concatenate_cols(vecs) for vecs in layer]  # each input is now a 2*D_h by len(input)

        else:  # todo add the bilstm option
            s0 = self.__leaf_lstm.initial_state()
            if self.__use_bilstm:
                bw_s0 = self.__bw_leaf_lstm.initial_state()
            layer = []
            for inp in inputs:
                h0 = dy.zeros(self.__D_x)
                c0 = dy.zeros(self.__D_x)
                sen = [dy.concatenate([h0, c0])]
                last_h, last_c = h0, c0
                if self.__use_bilstm:
                    last_bw_h, last_bw_c = h0, c0

                length = len(inp)
                for i in xrange(1, length):
                    lstm_input = [inp[i], last_h, last_c]
                    hc = s0.transduce(lstm_input)
                    h, c = hc[0:D_h], hc[D_h:2 * D_h]
                    last_h, last_c = h, c

                    if self.__use_bilstm:
                        bw_lstm_input = [inp[length - 1 - i], last_bw_h, last_bw_c]
                        bw_hc = bw_s0.transduce(bw_lstm_input)
                        bw_h, bw_c = bw_hc[0:D_h], bw_hc[D_h:2 * D_h]
                        h, c = dy.concatenate(h, bw_h), dy.concatenate(c, bw_c)
                        last_bw_h, last_bw_c = bw_h, bw_c

                    sen.append(dy.concatenate(h, c))
                layer.append(sen)

        max_len = max(map(lambda hc: hc.dim()[0][1], layer))
        single_zreo = np.array([[0]])
        while max_len > 1:
            batch_parents = []
            batch_y = []
            batch_y_st = []
            for sen in layer:
                n = sen.dim()[0][1]
                if n == 1:
                    batch_parents.append(sen)
                    batch_y.append(dy.inputTensor(single_zreo))
                    batch_y_st.append(dy.inputTensor([[1]]))
                    continue
                parents = self.__parents_of_layer(sen)  # all possible parents of pairs in layer
                batch_parents.append(parents)

                # creating v_1,...,v_M_t+1, Eq. (12) in the paper
                parents_scores_before = self.__parents_scores(parents)
                score_sum = dy.sum_elems(parents_scores_before)
                parents_scores = dy.cdiv(parents_scores_before, score_sum)

                if not test:
                    y = self.gumbel_softmax(parents_scores, self.__temperatue)
                else:
                    y = parents_scores
                batch_y.append(y)
                batch_y_st.append(self.__y_st_before_argmax(parents))
            # for's end

            dy.forward(batch_y_st)

            new_layer = []
            for i, (y, y_st_before) in enumerate(izip(batch_y, batch_y_st)):
                parents = batch_parents[i]
                if parents.dim()[0][1] == 1:  # sentence is already one node
                    new_layer.append(parents)
                    continue

                y_st_before = y_st_before.npvalue()
                n = y_st_before.shape[1]
                y_st = np.eye(n)
                amax = y_st_before.argmax(axis=1)
                y_st = y_st[amax]  # one-hot Straight Through (ST) vector
                y_st = dy.inputTensor(y_st[0])

                # in forward pass, uses the one-hot y_st, but backwards propagates to the gumbel-softmax vector, y
                y_hat = dy.nobackprop(y_st - y) + y

                Mt = layer[i].dim()[0][1]
                cumsum = self.cumsum(y_hat)  # c[i] = sum([y1, ..., yi])

                m_l = 1 - cumsum
                if Mt > 2:
                    m_r = dy.concatenate([dy.zeros(1), cumsum[0:Mt-2]])  # Mt - 2 = M(t+1) - 1
                else:
                    m_r = dy.zeros(1)
                m_p = y_hat

                M_l = dy.transpose(dy.concatenate_cols([m_l for _ in xrange(2 * D_h)]))
                M_r = dy.transpose(dy.concatenate_cols([m_r for _ in xrange(2 * D_h)]))
                M_p = dy.transpose(dy.concatenate_cols([m_p for _ in xrange(2 * D_h)]))

                new_r = dy.cmult(M_l, dy.select_cols(layer[i], range(Mt - 1)))  # lefts
                cmult_l = dy.cmult(M_r, dy.select_cols(layer[i], range(1, Mt)))
                new_r += cmult_l  # rights
                cmult_p = dy.cmult(M_p, parents)
                new_r += cmult_p  # parents
                new_layer.append(new_r)  # the new representation of the sentence

            layer = new_layer
            max_len = max(map(lambda hc: hc.dim()[0][1], layer))  # checking if all our sentences are one node
        # end of while

        return layer


class SNLIGumbelSoftmaxTreeLSTM:
    def __init__(self, D_h, D_x, D_c, mlp_hidden_layer_size, mlp_hidden_layers=1, temperatue=1.0, use_leaf_lstm=False,
                 lstm_layers=1, use_bilstm=False):
        """
        Ctor. This is an optimized batched version of the SimpleSNLIGumbelSoftmaxTreeLSTM.
        :param D_h: Size of inner representation vector.
        :param D_x: Size of input vectors.
        :param temperatue: The temperature for the gumbel softmax trick.
        :param use_leaf_lstm: If true uses an LSTM RNN to encode the leaf nodes.
        :param lstm_layers: How many layers the lstm will use, if used.
        :param use_bilstm: If true and use_leaf_lstm is true, uses a bilstm instead of a one way lstm.
        (This doubles D_h!!)
        :param D_c: Size of MLP layers
        :param mlp_hidden_layer_size: likewise
        :param mlp_hidden_layers: Number of hidden layers of MLP
        """
        self.__treeLSTM = GumbelSoftmaxTreeLSTM(D_h, D_x, temperatue, use_leaf_lstm, lstm_layers, use_bilstm)
        self.__D_c = D_c
        self.__D_h = D_h
        pc = self.__treeLSTM.pc
        self.ENTAILMENT, self.CONTRADICTION, self.NEUTRAL = 0, 1, 2
        self.__W_cl_f = [pc.add_parameters((1, D_c)), pc.add_parameters((1, D_c)), pc.add_parameters((1, D_c))]
        self.__b_cl_f = [pc.add_parameters(1), pc.add_parameters(1), pc.add_parameters(1)]

        # creates an mlp with n (n = {mlp_hidden_layers}) hidden layers and an input layer
        # input layer:
        # uni_init = dy.UniformInitializer(0.005)
        self.__mlp = [(pc.add_parameters((mlp_hidden_layer_size, 4 * D_h)),
                       pc.add_parameters(mlp_hidden_layer_size))]
        # n - 1 hidden layers:
        self.__mlp += [(pc.add_parameters((mlp_hidden_layer_size, mlp_hidden_layer_size)),
                        pc.add_parameters(mlp_hidden_layer_size)) for _ in xrange(mlp_hidden_layers - 1)]
        # last hidden layer to output:
        self.__mlp += [(pc.add_parameters((D_c, mlp_hidden_layer_size)),
                        pc.add_parameters(mlp_hidden_layer_size))]
        self.__mlp_activation = dy.rectify  # ReLu

    def get_parameter_collection(self):
        """
        :return: The model's dynet.ParameterCollection
        """
        return self.__treeLSTM.pc

    def __apply_mlp(self, f):
        """
        Applying the MLP to the input f, using the activation function (default is ReLu).
        :param f: Input to MLP
        :return: The MLP's output on f.
        """
        g = self.__mlp_activation
        x = f
        for pW, pb in self.__mlp[:-1]:
            W, b = dy.parameter(pW), dy.parameter(pb)
            x = g(W * x + b)
        pW, pb = self.__mlp[-1]
        W, b = dy.parameter(pW), dy.parameter(pb)
        return W * x + b

    def __call__(self, premises, hypotheses, test=False, use_dropout=False, dropout_prob=0.1):
        """
        For each sentence in teh batch, using the tree lstm represents the 2 sentences,
        then takes those represenation after processing and puts them into a MLP, then calculates probabilities for
        their relationship.
        :param premises: Batch of premise sentences (List of numpy vectors)
        :param hypotheses: Batch of hypothesis sentences (List of numpy vectors)
        :param test: If true doesn't uses the gumbel softmax but just chosses the max score.
        :param use_dropout: If true uses dropout on the input vectors, the MLP input and the output.
        :param dropout_prob: The probability for the dropout, if used.
        :return: List of vectors of probabilities for each class for each sentence.
        """
        dy.renew_cg()

        premises = [[dy.inputTensor(v) for v in premise] for premise in premises]
        hypotheses = [[dy.inputTensor(v) for v in hypothesis] for hypothesis in hypotheses]
        if use_dropout:  # dropout sentences
            premises = [[dy.dropout(v, dropout_prob) for v in premise] for premise in premises]
            hypotheses = [[dy.dropout(v, dropout_prob) for v in hypothesis] for hypothesis in hypotheses]

        hcs_pre = self.__treeLSTM(premises, renew_cg=False, test=test)
        hcs_hyp = self.__treeLSTM(hypotheses, renew_cg=False, test=test)
        D_h = self.__D_h
        zeroToD_h = range(D_h)
        batch_probs = []
        for hc_pre, hc_hyp in izip(hcs_pre, hcs_hyp):
            h_pre = dy.select_rows(hc_pre, zeroToD_h)
            h_hyp = dy.select_rows(hc_hyp, zeroToD_h)

            f = dy.concatenate([h_pre, h_hyp, dy.abs(h_pre - h_pre), dy.cmult(h_pre, h_hyp)])
            if use_dropout:
                f = dy.dropout(f, dropout_prob)
            a = self.__apply_mlp(f)
            if use_dropout:
                a = dy.dropout(a, dropout_prob)

            probs = [0, 0, 0]
            for i in [self.ENTAILMENT, self.CONTRADICTION, self.NEUTRAL]:
                W, b = dy.parameter(self.__W_cl_f[i]), dy.parameter(self.__b_cl_f[i])
                probs[i] = W * a + b
            probs = dy.concatenate(probs)
            batch_probs.append(dy.softmax(probs))
        return batch_probs

    def loss_on_batch(self, premises, hypotheses, batch_expected, use_dropout=False, dropout_prob=0.1):
        """
        Computes loss on the batch.
        :param premises: Batch of premise sentences (List of numpy vectors)
        :param hypotheses: Batch of hypothesis sentences (List of numpy vectors)
        :param batch_expected: Expected labels for each sentence
        :param use_dropout: If true uses dropout on the input vectors, the MLP input and the output.
        :param dropout_prob: The probability for the dropout, if used.
        :return: A list, where the i-th element is the loss of the sentences (premises[i], hypotheses[i])
        """
        batch_probs = self(premises, hypotheses, test=False, use_dropout=use_dropout, dropout_prob=dropout_prob)
        return [-dy.log(probs[expected]) for probs, expected in izip(batch_probs, batch_expected)]

    def predict_batch(self, premises, hypotheses):
        """
        Computes accuracy on the batch.
        :param premises: Batch of premise sentences (List of numpy vectors)
        :param hypotheses: Batch of hypothesis sentences (List of numpy vectors)
        :return: A list of the predicted class indexes for each (premise, hypothesis)
        """
        batch_probs = self(premises, hypotheses, test=True)
        return [probs.npvalue().argmax() for probs in batch_probs]
