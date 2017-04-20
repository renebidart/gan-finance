import sys
import os
from build_GAV_models import*


def main(data_file, out_loc, num_gen, batch_size, epochs, lr_gen=.01, lr_desc=0.0005, lr_both=0.0005): # use 50, 1600- 320steps*1000per batch
    import keras
    from keras.models import Sequential
    from keras.layers import Dropout, Flatten
    from keras.layers.core import Activation, Dense, Reshape
    from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import SGD
    from keras import backend as K
    import numpy as np 
    import random
    import pandas as pd

    def train(data_file, out_loc, batch_size, epochs, lr_gen, lr_desc, lr_both):
        batch_size = int(batch_size)
        epochs = int(epochs)

        ######### Initalize the models, and then combine them to create discriminator_on_generator
        generator = gen_model_stock(lr=lr_gen)
        discriminator = desc_model_stock(lr=lr_desc)
        discriminator_on_generator = build_gen_desc_stock(generator, discriminator,  lr=lr_both)
        discriminator.trainable = True

        # Save the training loss:
        desc_loss_hist=[]
        gen_loss_hist=[]

        ######### Train the model:
        ## Maybe it would be best to just make two data generators
        ## Really I just want to use the keras function to nicely output training progress
        data=pd.read_csv(data_file, sep=',').as_matrix().astype(float)[0:80000,1:] # leave some out just incase
        data=np.diff(data, axis=1)
        data=data[:,0:248]
        print 'data shape: ', data.shape
        for epoch in range(epochs):
            print 'epoch: ', epoch
            # randomize the data
            data = list(data)
            random.shuffle(data)
            data=np.array(data)
            print np.mean(data)
            data = data/(.5*data.ptp(0)) #is this the correct way to normalize???
            #data=np.expand_dims(data, axis=2)

            # See how many batches you'll need to cover the whole set 
            for index in range(int(data.shape[0]/batch_size)):
                # Generate noise
                noise = np.zeros((batch_size, 32))
                for i in range(batch_size):
                    noise[i, :] = np.random.uniform(0, 1, 32)
                # Now generate the random images:
                generated_data = generator.predict(noise, verbose=0)
                # get real images, and join everything together with the ys for the discriminator
                true_batch = data[index*batch_size:(index+1)*batch_size]
                true_batch=np.expand_dims(true_batch, axis=2)

                x_data = np.concatenate((true_batch, generated_data)).astype(float)                
                y_data=np.append(np.repeat(0, batch_size), np.repeat(1, batch_size), axis=0)
                y_data=np.eye(2)[y_data].astype(float)

                d_loss = discriminator.train_on_batch(x_data, y_data) # Is it better to randomize this???
                if index % 500 == 0:
                    desc_loss_hist.append(d_loss)
                    print("batch %d d_loss : %f" % (index, d_loss))

                ## Now train the generator:
                discriminator.trainable = False
                for i in range(batch_size):
                    noise[i, :] = np.random.uniform(0, 1, 32)
                y_noise=np.eye(2)[np.repeat(1, batch_size)].astype(float)
                g_loss = discriminator_on_generator.train_on_batch(noise, y_noise)
                discriminator.trainable = True
                if index % 500 == 0:
                    gen_loss_hist.append(g_loss)
                    print("batch %d g_loss           : %f" % (index, g_loss))
            
            generator_hist_file=os.path.join(str(out_loc)+'/'+'generator_hist')
            discriminator_hist_file=os.path.join(str(out_loc)+'/'+'discriminator_hist')
            np.save(generator_hist_file, desc_loss_hist)
            np.save(discriminator_hist_file, gen_loss_hist)

            generator_file=os.path.join(str(out_loc)+'/'+'generator')
            discriminator_file=os.path.join(str(out_loc)+'/'+'discriminator')
            generator.save_weights(generator_file, True)
            discriminator.save_weights(discriminator_file, True)
        

    # All done in the shit way with no variable input functions
    def generate(out_loc, num, only_good_ones=True):
        generated_samples=[]
        num=int(num)

        generator = gen_model_stock()
        generator_file=os.path.join(str(out_loc)+'/'+'generator')
        print generator_file
        generator.load_weights(generator_file)

        if only_good_ones: #take only the top 1 in every 10 generated images
            discriminator=desc_model_stock()
            discriminator_file=os.path.join(str(out_loc)+'/'+'discriminator')
            discriminator.load_weights(discriminator_file)
            for index in range(num):
                noise = np.zeros((10, 32))
                for i in range(10):
                    noise[i, :] = np.random.uniform(0, 1, 32)
                noise=np.reshape(noise, (10, 32))
                gen_data = generator.predict(noise, verbose=1)
                gen_data_pred = discriminator.predict(gen_data, verbose=1)[:,0] # hot-one, 1st col is p(true) 
                print gen_data


                # Now sort the images based on how good they were, and take the best one
                order_list=range(10)
                gen_data_pred, inds = (list(t) for t in zip(*sorted(zip(gen_data_pred, order_list))))
                idx = int(inds[9])
                good_gen_data = gen_data[idx, :, :]
                generated_samples.append(good_gen_data)

                # Whats going on?
                print 'Probability of true: ', gen_data_pred[0]


        else:
            for index in range(num):
                noise = np.random.uniform(0, 1, 32)
                noise=np.reshape(noise, (1, 32))
                data = generator.predict(noise, verbose=1)
                generated_samples.append(data)
        generated_samples=np.array(generated_samples)

        generated_samples=np.array(generated_samples)
        print 'shape of generated_samples: ', generated_samples.shape
        
        out_file =os.path.join(str(out_loc)+'/'+'generated_data')
        np.save(out_file, generated_samples)


    train(data_file, out_loc, batch_size=batch_size, epochs=epochs)
    generate(out_loc=out_loc, num=num_gen, only_good_ones=True)


if __name__ == "__main__":
    data_file = sys.argv[1]
    out_loc = sys.argv[2]
    num_gen = sys.argv[3]
    batch_size = sys.argv[4]
    epochs = sys.argv[5]

    main(data_file, out_loc, num_gen, batch_size, epochs)
