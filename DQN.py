# Modified version of
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, params):
        self.params = params
        self.network_name = 'qnet'
        self.sess = tf.compat.v1.Session()
        self.x = tf.compat.v1.placeholder('float', [None, params['width'],params['height'], 6],name=self.network_name + '_x')
        self.q_t = tf.compat.v1.placeholder('float', [None], name=self.network_name + '_q_t')
        self.actions = tf.compat.v1.placeholder("float", [None, 4], name=self.network_name + '_actions')
        self.rewards = tf.compat.v1.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.compat.v1.placeholder("float", [None], name=self.network_name + '_terminals')
        self.debug = params.get('debug', False)

        # Layer 1 (Convolutional)
        layer_name = 'conv1' ; size = 3 ; channels = 6 ; filters = 32 ; stride = 1
        self.w1 = tf.Variable(tf.random.normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o1 = tf.nn.relu(tf.add(self.c1,self.b1),name=self.network_name + '_'+layer_name+'_activations')

        # Layer 2 (Convolutional)
        layer_name = 'conv2' ; size = 3 ; channels = 32 ; filters = 64 ; stride = 1
        self.w2 = tf.Variable(tf.random.normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o2 = tf.nn.relu(tf.add(self.c2,self.b2),name=self.network_name + '_'+layer_name+'_activations')
        
        o2_shape = self.o2.get_shape().as_list()        

        # Layer 3 (Fully connected)
        layer_name = 'fc3' ; hiddens = 512 ; dim = o2_shape[1]*o2_shape[2]*o2_shape[3]
        self.o2_flat = tf.reshape(self.o2, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')
        self.w3 = tf.Variable(tf.random.normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat,self.w3),self.b3,name=self.network_name + '_'+layer_name+'_ips')
        self.o3 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations')
        
        # Add dropout to prevent overfitting
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name=self.network_name + '_dropout_keep_prob')
        self.o3_drop = tf.nn.dropout(self.o3, rate=1-self.keep_prob, name=self.network_name + '_dropout')

        # Layer 4
        layer_name = 'fc4' ; hiddens = 4 ; dim = 512
        self.w4 = tf.Variable(tf.random.normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.y = tf.add(tf.matmul(self.o3_drop,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')

        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), axis=1)
        
        # Use Huber loss for more stability with outliers
        self.delta = 1.0
        self.diff = tf.abs(self.yj - self.Q_pred)
        self.quadratic = tf.minimum(self.diff, self.delta)
        self.linear = self.diff - self.quadratic
        self.cost = tf.reduce_mean(0.5 * tf.square(self.quadratic) + self.delta * self.linear)
        
        if self.params['load_file'] is not None:
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]),name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # optimizer w/ gradient clipping
        optimizer = tf.compat.v1.train.AdamOptimizer(self.params['lr'])
        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # clip gradients to prevent explosion
        self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
        
        self.saver = tf.compat.v1.train.Saver(max_to_keep=0)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('loading checkpoint.')
            self.saver.restore(self.sess,self.params['load_file'])

        
    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r, self.keep_prob: 1.0}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        
        # Get max Q value for each next state
        q_t = np.amax(q_t, axis=1)
        
        if self.debug:
            print("Next state Q values:", q_t[:5])  # Print first 5 Q values
            
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r, self.keep_prob: 0.8}  # Apply dropout during training
        _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)
        
        if self.debug and np.isnan(cost):
            print("WARNING: NaN cost detected")
            print("Q values:", q_t)
            print("Rewards:", bat_r)
            
        return cnt, cost

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)
