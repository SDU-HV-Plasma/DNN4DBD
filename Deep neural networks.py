import tensorflow.compat.v1 as tf
import tensorflow as tf1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()
import numpy as np
import csv

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#Normalization

#0-0.25
dfn1 = pd.read_csv('Normalization 0-0.25.CSV',encoding='gbk')
xn1=dfn1[['Cycles','Voltage']]
x1_scaler=MinMaxScaler(feature_range=(0,1))
Xn1=x1_scaler.fit_transform(xn1)

#0.25-0.5
dfn2 = pd.read_csv('Normalization 0.25-0.5.CSV',encoding='gbk')
xn2=dfn2[['Cycles','Voltage']]
x2_scaler=MinMaxScaler(feature_range=(0,1))
Xn2=x2_scaler.fit_transform(xn2)

#0.5-0.75
dfn3 = pd.read_csv('Normalization 0.5-0.75.CSV',encoding='gbk')
xn3=dfn3[['Cycles','Voltage']]
x3_scaler=MinMaxScaler(feature_range=(0,1))
Xn3=x3_scaler.fit_transform(xn3)

#0.75-1
dfn4 = pd.read_csv('Normalization 0.75-1.CSV',encoding='gbk')
xn4=dfn4[['Cycles','Voltage']]
x4_scaler=MinMaxScaler(feature_range=(0,1))
Xn4=x4_scaler.fit_transform(xn4)

#Train set

df = pd.read_csv('Train set.CSV',encoding='gbk')

#0-0.25
a=df[df.Cycles>=0]
dfa=a[a.Cycles<=0.25]
xa=dfa[['Cycles','Voltage']]
xaa=x1_scaler.transform(xa)
xa_data=np.array(xaa,dtype='float32')
ya=dfa[['Current density']]
ya_data=np.array(ya,dtype='float32')

#0.25-0.5
b=df[df.Cycles>=0.25]
dfb=b[b.Cycles<=0.5]
xb=dfb[['Cycles','Voltage']]
xbb=x2_scaler.transform(xb)
xb_data=np.array(xbb,dtype='float32')
yb=dfb[['Current density']]
yb_data=np.array(yb,dtype='float32')

#0.5-0.75
c=df[df.Cycles>=0.5]
dfc=c[c.Cycles<=0.75]
xc=dfc[['Cycles','Voltage']]
xcc=x3_scaler.transform(xc)
xc_data=np.array(xcc,dtype='float32')
yc=dfc[['Current density']]
yc_data=np.array(yc,dtype='float32')

#0.75-1
d=df[df.Cycles>=0.75]
dfd=d[d.Cycles<=1]
xd=dfd[['Cycles','Voltage']]
xdd=x4_scaler.transform(xd)
xd_data=np.array(xdd,dtype='float32')
yd=dfd[['Current density']]
yd_data=np.array(yd,dtype='float32')

#Test set

dft = pd.read_csv('Test set 10000Hz 2450V.CSV',encoding='gbk')

#0-0.25
at=dft[dft.Cycles>=0]
dfat=at[at.Cycles<0.25]
xat=dfat[['Cycles','Voltage']]
xaat=x1_scaler.transform(xat)
xat_data=np.array(xaat,dtype='float32')
yat=dfat[['Current density']]
yat_data=np.array(yat,dtype='float32')

#0.25-0.5
bt=dft[dft.Cycles>=0.25]
dfbt=bt[bt.Cycles<0.5]
xbt=dfbt[['Cycles','Voltage']]
xbbt=x2_scaler.transform(xbt)
xbt_data=np.array(xbbt,dtype='float32')
ybt=dfbt[['Current density']]
ybt_data=np.array(ybt,dtype='float32')

#0.5-0.75
ct=dft[dft.Cycles>=0.5]
dfct=ct[ct.Cycles<0.75]
xct=dfct[['Cycles','Voltage']]
xcct=x3_scaler.transform(xct)
xct_data=np.array(xcct,dtype='float32')
yct=dfct[['Current density']]
yct_data=np.array(yct,dtype='float32')

#0.75-1
dt=dft[dft.Cycles>=0.75]
dfdt=dt[dt.Cycles<=1]
xdt=dfdt[['Cycles','Voltage']]
xddt=x4_scaler.transform(xdt)
xdt_data=np.array(xddt,dtype='float32')
ydt=dfdt[['Current density']]
ydt_data=np.array(ydt,dtype='float32')

# Input

xs1 = tf.placeholder(tf.float32, [None,2])
ys1 = tf.placeholder(tf.float32, [None,1])
xs2 = tf.placeholder(tf.float32, [None,2])
ys2 = tf.placeholder(tf.float32, [None,1])
xs3 = tf.placeholder(tf.float32, [None,2])
ys3 = tf.placeholder(tf.float32, [None,1])
xs4 = tf.placeholder(tf.float32, [None,2])
ys4 = tf.placeholder(tf.float32, [None,1])

# Hidden layer

l1 = add_layer(xs1, 2, 30, activation_function=tf1.nn.relu)
l2 = add_layer(l1, 30, 30, activation_function=tf1.tanh)
l3 = add_layer(l2, 30, 30, activation_function=tf1.tanh)
l4 = add_layer(l3, 30, 30, activation_function=tf1.sigmoid)

l5 = add_layer(xs2, 2, 30, activation_function=tf1.nn.relu)
l6 = add_layer(l5, 30, 30, activation_function=tf1.tanh)
l7 = add_layer(l6, 30, 30, activation_function=tf1.tanh)
l8 = add_layer(l7, 30, 30, activation_function=tf1.sigmoid)

l9 = add_layer(xs3, 2, 30, activation_function=tf1.nn.relu)
l10 = add_layer(l9, 30, 30, activation_function=tf1.tanh)
l11 = add_layer(l10, 30, 30, activation_function=tf1.tanh)
l12 = add_layer(l11, 30, 30, activation_function=tf1.sigmoid)

l13 = add_layer(xs4, 2, 30, activation_function=tf1.nn.relu)
l14 = add_layer(l13, 30, 30, activation_function=tf1.tanh)
l15 = add_layer(l14, 30, 30, activation_function=tf1.tanh)
l16 = add_layer(l15, 30, 30, activation_function=tf1.sigmoid)

# Output

prediction1 = add_layer(l4, 30, 1, activation_function=None)
prediction2 = add_layer(l8, 30, 1, activation_function=None)
prediction3 = add_layer(l12, 30, 1, activation_function=None)
prediction4 = add_layer(l16, 30, 1, activation_function=None)

# Train loss

loss1 = tf.reduce_mean(tf.square(ys1 - prediction1))
loss2 = tf.reduce_mean(tf.square(ys2 - prediction2))
loss3 = tf.reduce_mean(tf.square(ys3 - prediction3))
loss4 = tf.reduce_mean(tf.square(ys4 - prediction4))

train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss2)
train_step3 = tf.train.AdamOptimizer(0.0001).minimize(loss3)
train_step4 = tf.train.AdamOptimizer(0.0001).minimize(loss4)

# Computation loss

lossc1 = 100*tf.reduce_mean(tf.abs((ys1 - prediction1)/ys1))
lossc2 = 100*tf.reduce_mean(tf.abs((ys2 - prediction2)/ys2))
lossc3 = 100*tf.reduce_mean(tf.abs((ys3 - prediction3)/ys3))
lossc4 = 100*tf.reduce_mean(tf.abs((ys4 - prediction4)/ys4))

# Train

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    is_train = False
    is_mod = True
    is_mod2 = True
    saver = tf.train.Saver(max_to_keep=1)

    if is_train:
      if is_mod:
       if is_mod2:
        model_file1 = tf1.train.latest_checkpoint('save1/')
        saver.restore(sess, model_file1)
        for i in range(100001):
         sess.run(train_step1, feed_dict={xs1: xa_data, ys1: ya_data})
         if i % 100 == 0:
          print(sess.run(loss1, feed_dict={xs1: xa_data, ys1: ya_data}))

        saver.save(sess, 'save1/model1', global_step=i + 1)
       else:
           model_file2 = tf1.train.latest_checkpoint('save2/')
           saver.restore(sess, model_file2)
           for i in range(100001):
               sess.run(train_step2, feed_dict={xs2: xb_data, ys2: yb_data})
               if i % 100 == 0:
                   print(sess.run(loss2, feed_dict={xs2: xb_data, ys2: yb_data}))

           saver.save(sess, 'save2/model2', global_step=i + 1)
      else:
       if is_mod2:
        model_file3 = tf1.train.latest_checkpoint('save3/')
        saver.restore(sess, model_file3)
        for i in range(100001):
         sess.run(train_step3, feed_dict={xs3: xc_data, ys3: yc_data})
         if i % 100 == 0:
          print(sess.run(loss3, feed_dict={xs3: xc_data, ys3: yc_data}))

        saver.save(sess, 'save3/model3', global_step=i + 1)
       else:
           model_file4 = tf1.train.latest_checkpoint('save4/')
           saver.restore(sess, model_file4)
           for i in range(100001):
               sess.run(train_step4, feed_dict={xs4: xd_data, ys4: yd_data})
               if i % 100 == 0:
                   print(sess.run(loss4, feed_dict={xs4: xd_data, ys4: yd_data}))

           saver.save(sess, 'save4/model4', global_step=i + 1)

# Computation
    else:
        model_file1=tf1.train.latest_checkpoint('save1/')
        saver.restore(sess,model_file1)
        print(sess.run(lossc1, feed_dict={xs1: xat_data, ys1: yat_data}))
        with open("Current density 10000Hz 2450V.csv","w",newline='') as f:
         b_csv = csv.writer(f)
         b_csv.writerows(sess.run(prediction1, feed_dict={xs1: xat_data}))

         model_file2 = tf1.train.latest_checkpoint('save2/')
         saver.restore(sess, model_file2)
         print(sess.run(lossc2, feed_dict={xs2: xbt_data, ys2: ybt_data}))
         b_csv.writerows(sess.run(prediction2, feed_dict={xs2: xbt_data}))

         model_file3 = tf1.train.latest_checkpoint('save3/')
         saver.restore(sess, model_file3)
         b_csv.writerows(sess.run(prediction3, feed_dict={xs3: xct_data}))
         print(sess.run(lossc3, feed_dict={xs3: xct_data, ys3: yct_data}))

         model_file4 = tf1.train.latest_checkpoint('save4/')
         saver.restore(sess, model_file4)
         b_csv.writerows(sess.run(prediction4, feed_dict={xs4: xdt_data}))
         print(sess.run(lossc4, feed_dict={xs4: xdt_data, ys4: ydt_data}))