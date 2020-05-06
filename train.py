from cifar import Cifar
from tqdm import tqdm
import tensorflow as tf
import model
import helper

learning_rate = 0.001
batch_size = 16
no_of_epochs = 10
dropout_rate = 0.8

y = tf.compat.v1.placeholder(tf.float32, [None, model.n_classes])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.out, labels=y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.compat.v1.summary.histogram("cost", cost)          # <- Added this summary of cost
tf.compat.v1.summary.histogram("accuracy", accuracy)  # <- Added this summary of accuracy

cifar = Cifar(batch_size=batch_size)

init = tf.initialize_all_variables()

saver = tf.compat.v1.train.Saver()
i = 0

with tf.compat.v1.Session() as sess:
    sess.run(init)
    
    # Create writer, which writes to ./logs folder  
    writer = tf.compat.v1.summary.FileWriter( './logs/', sess.graph)

    for epoch in range(no_of_epochs):
        for batch in tqdm(cifar.batches, desc="Epoch {}".format(epoch), unit="batch"):
            inp, out = helper.transform_to_input_output(batch, dim=model.n_classes)

            sess.run([optimizer], feed_dict={model.input_images: inp, y: out, model.dropout: dropout_rate})


        merge = tf.compat.v1.summary.merge_all()
        acc, loss, summary = sess.run([accuracy, cost, merge], feed_dict={model.input_images: inp, y: out, model.dropout: 1.})
        writer.add_summary(summary, i)
        i+=1

        print("Train Acc: {} Loss: {}".format(acc, loss))

        # Test the model
        inp_test, out_test = helper.transform_to_input_output(cifar.test_set, dim=model.n_classes)

        test_acc = sess.run([accuracy], feed_dict={model.input_images: inp_test,y: out_test, model.dropout: 1.})
        print("Test Acc: {}".format(test_acc))

        saver.save(sess, "saved_model/alexnet.ckpt")
