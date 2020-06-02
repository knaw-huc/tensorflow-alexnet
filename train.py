from cifar import Cifar
from rvl_cdip import RvlCdip
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from model import Model
import helper
import os
import argparse


arg_parser = argparse.ArgumentParser(description='Classify document type')
arg_parser.add_argument('--learning_rate', type=float, default=0.0001, dest='learning_rate', help='default is 0.0001')
arg_parser.add_argument('--batch_size', type=int, default=1000, dest='batch_size', help='default is 1000')
arg_parser.add_argument('--epochs', type=int, default=200, dest='epochs', help='default is 200')
arg_parser.add_argument('--dropout_rate', type=float, default=0.1, dest='dropout_rate', help='default is 0.1')
arg_parser.add_argument('--grayscale', type=bool, default=True, dest='grayscale', help='default is True')
arg_parser.add_argument('--model_path', type=str, default='saved_model/alexnet.ckpt', dest='model_path', help='path to store the model (default is saved_model/alexnet.ckpt)')
arg_parser.add_argument('--logs_path', type=str, default='logs', dest='logs_path', help='path to store the logs (default is logs)')
arg_parser.add_argument('--fc_size', type=int, default=1024, dest='fc_size', help='size of fully connect layers of neural network (default is 1024)')
arg_parser.add_argument('--image_size', type=int, default=227, dest='image_size', help='size of input images (default is 227)')
arg_parser.add_argument('--gpu', type=str, default="0", dest='gpu', help='which gpu to use default "0"')


args = arg_parser.parse_args()

learning_rate = args.learning_rate
batch_size = args.batch_size
no_of_epochs = args.epochs 
dropout_rate = args.dropout_rate
fc_size = args.fc_size
image_size = args.image_size

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

session_file = args.model_path


# dataset = Cifar(batch_size=batch_size)
dataset = RvlCdip(batch_size=batch_size, grayscale=args.grayscale, image_size=image_size)
n_classes = dataset.number_of_classes

model = Model(image_size=image_size, n_classes=n_classes, fc_size=fc_size)

y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
#model.image_size = image_size
#model.n_classes = n_classes
#model.fc_size = fc_size
#print("model.image_size", model.image_size)
#print("model.n_classes", model.n_classes)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.out, labels=y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.compat.v1.summary.histogram("cost", cost)          # <- Added this summary of cost
tf.compat.v1.summary.histogram("accuracy", accuracy)  # <- Added this summary of accuracy


init = tf.compat.v1.initialize_all_variables()

saver = tf.compat.v1.train.Saver()
i = 0
epoch_of_last_test_improvement = -1
best_test_loss = 1000 
with tf.compat.v1.Session() as sess:
    if os.path.exists(session_file):
        print("load model")
        saver.restore(sess, session_file)
    else:
        sess.run(init)
    # Create writer, which writes to ./logs folder  
    train_writer = tf.compat.v1.summary.FileWriter( args.logs_path + '/train/', sess.graph)
    val_writer = tf.compat.v1.summary.FileWriter( args.logs_path + '/val/', sess.graph)
    for epoch in range(no_of_epochs):
        no_of_batches = len(dataset.batches)
        merge = tf.compat.v1.summary.merge_all()
        for batch_no in tqdm(range(no_of_batches), desc="Epoch {}".format(epoch), unit="batch"):
            batch = dataset.batch(batch_no)
            inp, out = helper.transform_to_input_output(batch, dim=dataset.number_of_classes)

            sess.run([optimizer, merge], feed_dict={model.input_images: inp, y: out, model.dropout: dropout_rate})



        acc, loss, summary = sess.run([accuracy, cost, merge], feed_dict={model.input_images: inp, y: out, model.dropout: 0.})
        train_writer.add_summary(summary, i)
        i+=1

        print("Train Acc: {} Loss: {}".format(acc, loss))

        # Test the model
        inp_test, out_test = helper.transform_to_input_output(dataset.test_set, dim=model.n_classes)

        test_acc, test_loss, test_summary = sess.run([accuracy, cost, merge], feed_dict={model.input_images: inp_test,y: out_test, model.dropout: 0.})
        val_writer.add_summary(test_summary, i)
        print("Test Acc: {} Loss: {}".format(test_acc, test_loss))
#        print("Test Acc: {}".format(test_acc))

        if test_loss < best_test_loss:
            epoch_of_last_test_improvement = epoch
            best_test_loss = test_loss
            print("save checkpoint")
            saver.save(sess, session_file)

        if (epoch - epoch_of_last_test_improvement) >= 20:
            break
