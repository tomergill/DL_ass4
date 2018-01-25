import json
from time import time
import dynet as dy
import numpy as np
from sys import argv
import GumbelSoftmaxTreeLSTM as gst
from random import shuffle

UNKNOWN = "UNK"


def read_glove(file_name):
    W2NV = {}
    for line in file(file_name):
        line = line.split()
        word = line.pop(0)
        W2NV[word] = np.array(map(float, line))
    return W2NV


def read_snli_data_file(file_name):
    sentences_and_tags = []
    for line in file(file_name):
        js = json.loads(line[:-1])
        if js["gold_label"] == "-":
            continue
        sentences_and_tags.append((js["sentence1"], js["sentence2"], js["gold_label"]))
    return sentences_and_tags


def dataset_to_numerical_data(sentences_and_tags, W2NV, model):
    """

    :param sentences_and_tags:
    :param W2NV:
    :type model: SimpleSNLIGumbelSoftmaxTreeLSTM
    :param model:
    :return:
    """
    T2I = {"entailment": model.ENTAILMENT, "neutral": model.NEUTRAL, "contradiction": model.CONTRADICTION}
    return [([W2NV[w] if w in W2NV else W2NV[UNKNOWN] for w in sen1],
             [W2NV[w] if w in W2NV else W2NV[UNKNOWN] for w in sen2],
             T2I[tag]) for sen1, sen2, tag in sentences_and_tags]


def accuracy_on(model, data):
    """

    :type model: SimpleSNLIGumbelSoftmaxTreeLSTM
    :param model:
    :param data:
    :return:
    """
    good = 0.0
    shuffle(data)
    for sen1, sen2, tag in data:
        prediction = model.predict(sen1, sen2)
        if prediction == tag:
            good += 1.0
    return good / len(data)


def train_on(model, trainer, data, dev_data, epochs, dropout_p=0.0, print_every=50000):
    """

    :type model: SimpleSNLIGumbelSoftmaxTreeLSTM
    :param model:
    :param data:
    :param dev_data:
    :param epochs:
    :param dropout_p:
    :return:
    """
    use_dropout = dropout_p > 0.0
    write_to_file = ["epoch,average_loss,total_time,dev_accuracy"]
    print "+-------+--------------+------------+--------------+"
    print "| Epoch | Average_Loss | Total_Time | Dev_Accuracy |"
    print "+-------+--------------+------------+--------------+"
    for i in xrange(epochs):
        shuffle(data)
        total_loss = 0.0
        start_time = time()
        for j, (pre, hyp, tag) in enumerate(data):
            loss = model.loss_on(pre, hyp, tag, use_dropout, dropout_p)
            total_loss += loss.value()
            loss.backward()
            trainer.update()

            if j % print_every == print_every - 1:
                acc = accuracy_on(model, dev_data)
                write_to_file.append("{},{},{},{}".format(epochs, total_loss, time() - start_time, acc))
                print "| {:>5} | {:12f} | {:8f} s | {:10f} % |".format(epochs, total_loss, time() - start_time, acc*100)
                print "+-------+--------------+------------+--------------+"
    return write_to_file


def accuracy_on_batch(model, data, batch_size=128):
    """

    :type model: gst.SNLIGumbelSoftmaxTreeLSTM
    :param model:
    :param data:
    :param batch_size:
    :return:
    """
    good = 0.0
    for i in xrange(0, len(data), batch_size):
        mini_batch = data[i:i + batch_size]
        premises, hypotheses, tags = zip(*mini_batch)
        premises, hypotheses, tags = list(premises), list(hypotheses), list(tags)
        batch_preds = model.predict_batch(premises, hypotheses)
        good += reduce(lambda total, (pred, expected): total + (1.0 if pred == expected else 0.0),
                       zip(batch_preds, tags), initializer=0.0)
    return good / len(data)


def train_on_with_batches(model, trainer, data, dev_data, epochs, dropout_p=0.0, print_every=50000, batch_size=128):
    """

    :type model: gst.SNLIGumbelSoftmaxTreeLSTM
    :param model:
    :param trainer:
    :param data:
    :param dev_data:
    :param epochs:
    :param dropout_p:
    :param print_every:
    :param batch_size:
    :return:
    """
    last = print_every - 1
    use_dropout = dropout_p > 0.0
    write_to_file = ["epoch,average_loss,total_time,dev_accuracy"]
    print "+-------+--------------+------------+--------------+"
    print "| Epoch | Average_Loss | Total_Time | Dev_Accuracy |"
    print "+-------+--------------+------------+--------------+"
    for i in xrange(epochs):
        shuffle(data)
        total = 0
        total_loss = 0.0
        start_time = time()
        for j in xrange(0, len(data), batch_size):
            mini_batch = data[j:j + batch_size]
            premises, hypotheses, tags = zip(*mini_batch)
            premises, hypotheses, tags = list(premises), list(hypotheses), list(tags)
            batch_losses = model.loss_on_batch(premises, hypotheses, tags, use_dropout, dropout_p)
            loss = dy.esum(batch_losses)
            total_loss += loss.value()
            total += batch_size
            loss.backward()
            trainer.update()

            if total % print_every == last:
                acc = accuracy_on_batch(model, dev_data, batch_size)
                write_to_file.append("{},{},{},{}".format(epochs, total_loss, time() - start_time, acc))
                print "| {:>5} | {:12f} | {:8f} s | {:10f} % |".format(epochs, total_loss / total, time() - start_time, acc*100)
                print "+-------+--------------+------------+--------------+"
    return write_to_file


def main():
    files_name = "snli_1.0_{}.jsonl"
    use_leaf_lstm = False  # -lstm
    use_leaf_bilstm = False  # -bilstm
    glove_file = "glove.840B.300d.txt"
    use_simple = False  # -simple

    if len(argv) > 1:
        for arg in argv[1:]:
            if arg == "-lstm":
                use_leaf_lstm = True
            if arg == "-bilstm":
                use_leaf_lstm = True
                use_leaf_bilstm = True
            if arg == "-simple":
                use_simple = True

    print "Starting reading GloVe file..."
    start = time()
    W2NV = read_glove(glove_file)
    D_x = W2NV[","].shape[0] if len(W2NV[","].shape) == 1 else W2NV[","].shape[1]
    print "Finished reading GloVe in {} seconds.\nStarting reading SNLI data sets...".format(time() - start)
    
    start = time()
    train_set = read_snli_data_file(files_name.format("train"))
    dev_set = read_snli_data_file(files_name.format("dev"))  # + read_snli_data_file(files_name.format("test"))
    print "Finished reading SNLI data sets in {} seconds.".format(time() - start)

    # parameters
    D_h = 300
    D_c = 1024
    mlp_hid_dim = D_c
    dropout_probability = 0.1
    epochs = 1

    if use_simple:
        model = gst.SimpleSNLIGumbelSoftmaxTreeLSTM(D_h, D_x, D_c, mlp_hid_dim, use_leaf_lstm=use_leaf_lstm,
                                                    use_bilstm=use_leaf_bilstm)
    else:
        model = gst.SNLIGumbelSoftmaxTreeLSTM(D_h, D_x, D_c, mlp_hid_dim, use_leaf_lstm=use_leaf_lstm,
                                              use_bilstm=use_leaf_bilstm)
    trainer = dy.AdamTrainer(model.get_parameter_collection())
    # todo remove size limits
    TRAIN, DEV = dataset_to_numerical_data(train_set, W2NV, model)[:10000], dataset_to_numerical_data(dev_set, W2NV, model)[:1000]

    print "##################################################"
    print "#\tWords in GloVe vocab: {}".format(len(W2NV))
    print "#\tGloVe embedded vectors dimension (D_x): {}".format(D_x)
    print "#\tHidden Dimension (D_h): {}".format(D_h)
    print "#\tD_c: {}".format(D_c)
    print "#\tMLP hidden dimension: {}".format(mlp_hid_dim)
    print "#\tDropout probability: {}".format(dropout_probability)
    print "#\tEpochs: {}".format(epochs)
    print "#\tTrain set size: {}".format(len(TRAIN))
    print "#\tDev set size: {}".format(len(DEV))
    print "#\tLeaf encoding: {}".format(("BiLSTM" if use_leaf_bilstm else "LSTM") if use_leaf_lstm else "Linear Layer")
    print "##################################################\n"

    if use_simple:
        write_to_file = train_on(model, trainer, TRAIN, DEV, epochs, dropout_p=dropout_probability)
    else:  # todo remove print_every=1000
        write_to_file = train_on_with_batches(model, trainer, TRAIN, DEV, epochs, dropout_probability, print_every=1000)

    output_file = open("log.csv", "w")
    for line in write_to_file:
        output_file.write(line + "\n")
    output_file.close()


if __name__ == '__main__':
    main()
