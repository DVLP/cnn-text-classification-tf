#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import sys
import data_helpers
from handler import MessageHandler
from tensorflow.contrib import learn

# Parameters
# ==================================================

checkpoint_dir = "./runs/1490234541/checkpoints"

# Map data into vocabulary
vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

print("\nEvaluating...\n", flush=True)

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    # Leave this in case we want debug data
    # session_conf = tf.ConfigProto(log_device_placement=True)
    # sess = tf.Session(config=session_conf)
    sess = tf.Session()
    with sess.as_default():
        handler = MessageHandler(sess, checkpoint_file, graph, vocab_processor)
        print("READY", flush=True)

        while True:
            line = sys.stdin.readline()

            if not line:
                pass

            handler.handle(line)