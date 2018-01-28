Just put trainTree.py, GumbelSoftmaxTreeLSTM.py, the GloVe 300D data and the snli_1.0_train/dev.jsonl files in the same
 place and run.
Arguments to the program can be: (order doesn't matter)
    -lstm : Uses a LSTM RNN to encode the leaf nodes instead of a liniear layer.
    -bilstm : Uses a biLSTM RNN to encode the leaf nodes instead of a liniear layer. (Doubles the size of data).
    -simple : Uses the non-batched non-optimized version of the model.
