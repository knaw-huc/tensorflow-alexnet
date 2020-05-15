from cifar import Cifar
from rvl_cdip import RvlCdip
from tqdm import tqdm
import tensorflow as tf
import model
# import model_example as model
import helper
import os

learning_rate = 0.001
# learning_rate = 0.00000001
batch_size = 100
no_of_epochs = 10
dropout_rate = 0.8

os.environ["CUDA_VISIBLE_DEVICES"]="1"

session_file = "saved_model/alexnet.ckpt"

# dataset = Cifar(batch_size=batch_size)
dataset = RvlCdip(batch_size=batch_size)
y = tf.compat.v1.placeholder(tf.float32, [None, dataset.number_of_classes])
print("model.image_size", model.image_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.out, labels=y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.compat.v1.summary.histogram("cost", cost)          # <- Added this summary of cost
tf.compat.v1.summary.histogram("accuracy", accuracy)  # <- Added this summary of accuracy


init = tf.compat.v1.initialize_all_variables()

saver = tf.compat.v1.train.Saver()
i = 0

with tf.compat.v1.Session() as sess:
    if os.path.exists("./saved_model"):
        print("load model")
        saver.restore(sess, session_file)
    else:
        sess.run(init)
    # Create writer, which writes to ./logs folder  
    writer = tf.compat.v1.summary.FileWriter( './logs/', sess.graph)

    for epoch in range(no_of_epochs):
        no_of_batches = len(dataset.batches)
        for batch_no in tqdm(range(no_of_batches), desc="Epoch {}".format(epoch), unit="batch"):
            batch = dataset.batch(batch_no)
            inp, out = helper.transform_to_input_output(batch, dim=dataset.number_of_classes)

            sess.run([optimizer], feed_dict={model.input_images: inp, y: out, model.dropout: dropout_rate})


        merge = tf.compat.v1.summary.merge_all()
        acc, loss, summary = sess.run([accuracy, cost, merge], feed_dict={model.input_images: inp, y: out, model.dropout: 1.})
        writer.add_summary(summary, i)
        i+=1

        print("Train Acc: {} Loss: {}".format(acc, loss))

        # Test the model
        no_test_sets = len(dataset.test_sets)
        total_test_acc = 0
        for test_no in tqdm(range(no_test_sets), desc="Epoch {}".format(epoch), unit="test set"):
            inp_test, out_test = helper.transform_to_input_output(dataset.test_set(test_no), dim=model.n_classes)

            test_acc = sess.run([accuracy], feed_dict={model.input_images: inp_test,y: out_test, model.dropout: 1.})
            total_test_acc += test_acc[0]
        print("Test Acc: {}".format(total_test_acc / no_test_sets))

        print("save checkpoint")
        saver.save(sess, session_file)
