import sys
import csv
import os
import data_helpers
import tensorflow as tf
import numpy as np

class MessageHandler(object):
    def __init__(self, sess, checkpoint_file, checkpoint_dir, graph, x_test, y_test, x_raw, batch_size):
        self.sess = sess
        self.checkpoint_file = checkpoint_file
        self.checkpoint_dir = checkpoint_dir
        self.graph = graph
        self.x_test = x_test
        self.y_test = y_test
        self.x_raw = x_raw
        self.batch_size = batch_size

    def handle(self, line):
        line = line[:-1]  # Remove newline from line end
        print("String {} to be handled".format(line))

        if line == "calculate":
            self.calculate()

    def calculate(self):
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
        batches = data_helpers.batch_iter(list(self.x_test), self.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self.sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        # Print accuracy if self.y_test is defined
        if self.y_test is not None:
            correct_predictions = float(sum(all_predictions == self.y_test))
            print("Total number of test examples: {}".format(len(self.y_test)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(self.y_test))))

        # Save the evaluation to a csv
        predictions_human_readable = np.column_stack((np.array(self.x_raw), all_predictions))
        out_path = os.path.join(self.checkpoint_dir, "..", "prediction.csv")
        print("Saving evaluation to {0}".format(out_path))
        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)