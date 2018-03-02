import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import tensorflow as tf

import load_vgg_sol
import utils

def setup():
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('outputs')

class StyleTransfer(object):
    def __init__(self, content_img, style_img, img_width, img_height):
        '''
        img_width and img_height are the dimensions we expect from the generated image.
        We will resize input content image and input style image to match this dimension.
        Feel free to alter any hyperparameter here and see how it affects your training.
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.content_img = utils.get_resized_image(content_img, img_width, img_height)
        self.style_img = utils.get_resized_image(style_img, img_width, img_height)
        self.initial_img = utils.generate_noise_image(self.content_img, img_width, img_height)

        ###############################
        ## TO DO
        ## create global step (gstep) and hyperparameters for the model
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.content_w = 0.01
        self.style_w = 1
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0] 
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.lr = 2.0
        ###############################

    def create_input(self):
        '''
        We will use one input_img as a placeholder for the content image, 
        style image, and generated image, because:
            1. they have the same dimension
            2. we have to extract the same set of features from them
        We use a variable instead of a placeholder because we're, at the same time, 
        training the generated image to get the desirable result.

        Note: image height corresponds to number of rows, not columns.
        '''
        with tf.variable_scope('input') as scope:
            self.input_img = tf.get_variable('in_img', 
                                        shape=([1, self.img_height, self.img_width, 3]),
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
    def load_vgg(self):
        '''
        Load the saved model parameters of VGG-19, using the input_img
        as the input to compute the output at each layer of vgg.

        During training, VGG-19 mean-centered all images and found the mean pixels
        to be [123.68, 116.779, 103.939] along RGB dimensions. We have to subtract
        this mean from our images.

        '''
        self.vgg = load_vgg_sol.VGG(self.input_img)
        self.vgg.load()
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    def _content_loss(self, P, F):
        ''' Calculate the loss between the feature representation of the
        content image and the generated image.
        
        Inputs: 
            P: content representation of the content image
            F: content representation of the generated image
            Read the assignment handout for more details

            Note: Don't use the coefficient 0.5 as defined in the paper.
            Use the coefficient defined in the assignment handout.
        '''
        # self.content_loss = None
        ###############################
        ## TO DO
        self.content_loss = tf.reduce_sum((F - P) ** 2) / (4.0 * P.size)
        ###############################
    
    def _gram_matrix(self, F, N, M):
        """ Create and return the gram matrix for tensor F
            Hint: you'll first have to reshape F
        """
        ###############################
        ## TO DO
        F = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(F), F)
        ###############################

    def _single_style_loss(self, a, g):
        """ Calculate the style loss at a certain layer
        Inputs:
            a is the feature representation of the style image at that layer
            g is the feature representation of the generated image at that layer
        Output:
            the style loss at a certain layer (which is E_l in the paper)

        Hint: 1. you'll have to use the function _gram_matrix()
            2. we'll use the same coefficient for style loss as in the paper
            3. a and g are feature representation, not gram matrices
        """
        ###############################
        ## TO DO
        N = a.shape[3] # number of filters
        M = a.shape[1] * a.shape[2] # height times width of the feature map
        A = self._gram_matrix(a, N, M)
        G = self._gram_matrix(g, N, M)
        return tf.reduce_sum((G - A) ** 2 / ((2 * N * M) ** 2))
        ###############################

    def _style_loss(self, A):
        """ Calculate the total style loss as a weighted sum 
        of style losses at all style layers
        Hint: you'll have to use _single_style_loss()
        """
        n_layers = len(A)
        E = [self._single_style_loss(A[i], getattr(self.vgg, self.style_layers[i])) for i in range(n_layers)]
        
        ###############################
        ## TO DO
        self.style_loss = sum([self.style_layer_w[i] * E[i] for i in range(n_layers)])
        ###############################

    def losses(self):
        with tf.variable_scope('losses') as scope:
            with tf.Session() as sess:
                # assign content image to the input variable
                sess.run(self.input_img.assign(self.content_img)) 
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)

            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])                              
            self._style_loss(style_layers)

            ##########################################
            ## TO DO: create total loss. 
            ## Hint: don't forget the weights for the content loss and style loss
            self.total_loss = self.content_w * self.content_loss + self.style_w * self.style_loss
            ##########################################

    def optimize(self):
        ###############################
        ## TO DO: create optimizer
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss,
                                                            global_step=self.gstep)
        ###############################

    def create_summary(self):
        ###############################
        ## TO DO: create summaries for all the losses
        ## Hint: don't forget to merge them
        with tf.name_scope('summaries'):
            tf.summary.scalar('content loss', self.content_loss)
            tf.summary.scalar('style loss', self.style_loss)
            tf.summary.scalar('total loss', self.total_loss)
            self.summary_op = tf.summary.merge_all()
        ###############################


    def build(self):
        self.create_input()
        self.load_vgg()
        self.losses()
        self.optimize()
        self.create_summary()

    def train(self, n_iters):
        skip_step = 1
        with tf.Session() as sess:
            
            ###############################
            ## TO DO: 
            ## 1. initialize your variables
            ## 2. create writer to write your graph
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('graphs/style_stranfer', sess.graph)
            ###############################
            sess.run(self.input_img.assign(self.initial_img))


            ###############################
            ## TO DO: 
            ## 1. create a saver object
            ## 2. check if a checkpoint exists, restore the variables
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/style_transfer/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            ##############################

            initial_step = self.gstep.eval()
            
            start_time = time.time()
            for index in range(initial_step, n_iters):
                if index >= 5 and index < 20:
                    skip_step = 10
                elif index >= 20:
                    skip_step = 20
                
                sess.run(self.opt)
                if (index + 1) % skip_step == 0:
                    ###############################
                    ## TO DO: obtain generated image, loss, and summary
                    gen_image, total_loss, summary = sess.run([self.input_img,
                                                                self.total_loss,
                                                                self.summary_op])

                    ###############################
                    
                    # add back the mean pixels we subtracted before
                    gen_image = gen_image + self.vgg.mean_pixels 
                    writer.add_summary(summary, global_step=index)
                    print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                    print('   Loss: {:5.1f}'.format(total_loss))
                    print('   Took: {} seconds'.format(time.time() - start_time))
                    start_time = time.time()

                    filename = 'outputs/%d.png' % (index)
                    utils.save_image(filename, gen_image)

                    if (index + 1) % 20 == 0:
                        ###############################
                        ## TO DO: save the variables into a checkpoint
                        saver.save(sess, 'checkpoints/style_stranfer/style_transfer', index)
                        ###############################

if __name__ == '__main__':
    setup()
    machine = StyleTransfer('content/deadpool.jpg', 'styles/guernica.jpg', 333, 250)
    machine.build()
    machine.train(300)