import tensorflow as tf

class Network:
    def __init__(self,keep_prob=0.8):
        self.keep_prob = keep_prob
    def vgg(self,net,is_training=True):
        net = tf.contrib.layers.conv2d(net,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv1')
        
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool1')
        
        net = tf.contrib.layers.conv2d(net,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv2')
        
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool2')

        net = tf.contrib.layers.conv2d(net,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv3')
        
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool3')
     
        net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv4')
 
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool4')

        net = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv5')
 
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool5')
        net = tf.contrib.layers.flatten(net, scope='flatten')
        ## funcl layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=256,activation_fn=tf.nn.relu, scope='fully_connected1')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout1')
        ## func2 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=128,activation_fn=tf.nn.relu, scope='fully_connected2')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout2')
        ## func3 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=2,activation_fn=None, scope='logits')

        return net
    def ResNet(net, training=False):
        net = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(5,5),stride=(2,2),padding='SAME',activation_fn=tf.nn.relu, scope='Conv1') # size = (75, 75)
        net = proposed_residual(net, name='block1', block_num=2, size='SAME', is_training=training)   # size = (75, 75)
        net = proposed_residual(net, name='block2',block_num=2, size='SAME', is_training=training)
        net = proposed_residual(net, name='block3',block_num=2, size='SAME', is_training=training)

        net = proposed_residual(net, name='block4', block_num=2, size='VALID', is_training=training)    # size = (38, 38)
        net = proposed_residual(net, name='block5', block_num=2, size='SAME', is_training=training)
        net = proposed_residual(net, name='block6', block_num=2, size='SAME', is_training=training)
        net = proposed_residual(net, name='block7', block_num=2, size='SAME', is_training=training)

        net = proposed_residual(net, name='block8', block_num=2, size='VALID', is_training=training)    # size = (19, 19)
        net = proposed_residual(net, name='block9', block_num=2, size='SAME', is_training=training)
        net = proposed_residual(net, name='block10', block_num=2, size='SAME', is_training=training)
        net = proposed_residual(net, name='block11', block_num=2, size='SAME', is_training=training)
        net = proposed_residual(net, name='block12', block_num=2, size='SAME', is_training=training)
        net = proposed_residual(net, name='block13', block_num=2, size='SAME', is_training=training)

        net = proposed_residual(net, name='block14', block_num=2, size='VALID', is_training=training)    # size = (10, 10)
        net = proposed_residual(net, name='block15', block_num=2, size='SAME', is_training=training)
        net = proposed_residual(net, name='block16', block_num=2, size='SAME', is_training=training)

        net = tf.contrib.layers.avg_pool2d(net, kernel_size=(10,10), stride=(2,2),padding='VALID',scope='AVG')

        net = tf.contrib.layers.flatten(net, scope='flatten')

        net = tf.contrib.layers.fully_connected(net,num_outputs=60,activation_fn=None, scope='Layer')
        
        return net
