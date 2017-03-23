import os
import data_helpers
import tensorflow as tf
import numpy as np

class MessageHandler(object):
    def __init__(self, sess, checkpoint_file, graph, vocab_processor):
        self.sess = sess
        self.checkpoint_file = checkpoint_file
        self.graph = graph
        self.vocab_processor = vocab_processor

    def handle(self, line):
        line = line[:-1]  # Remove newline from line end
        print("String {} to be handled".format(line))

        if line.startswith("calculate"):
            self.calculate(line[len("calculate") + 1:])

    def calculate(self, text):
        x_raw = [data_helpers.clean_str(text)]
        x_test = np.array(list(self.vocab_processor.transform(x_raw)))

        print("Input string: \"{}\"".format(text))
        print("Calculating")

        # Load the saved meta self.graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
        saver.restore(self.sess, self.checkpoint_file)

        # Get the placeholders from the self.graph by name
        input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        # input_y = self.graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        all_predictions = self.sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})

        # Save the evaluation to a csv
        predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
        print("Output: {}".format(predictions_human_readable[0]))