# Tensorboard-own-image-data-image-features-embedding-visualization
Learn how to visualize your own image data or features on Tensorboard Embedding Visualizer.
The video tutorial for the same is available at: https://www.youtube.com/watch?v=CNR7Wu7g2aY

# Libraries:
    keras-1.2.1
    Tensorflow -1.0.1
    python-3.5

# Runing the embedding visualization using the logs given in this repository
To run the embeddings already provided in embedding-logs. Download all the files.

    In the embedding-logs/checkpoint
    
    model_checkpoint_path: "D:\\Technical_works\\tensorflow\\own-data-embedding-visualization-vgg-16/embedding-     logs\\images_4_classes.ckpt"
    all_model_checkpoint_paths: "D:\\Technical_works\\tensorflow\\own-data-embedding-visualization-vgg-16/embedding-logs\\images_4_classes.ckpt"
    
    Change the model_checkpoint_path and all_model_checkpoint_paths to your path

    In the embedding-logs/projector_config.pbtxt
       
            embeddings {
        tensor_name: "features:0"
        metadata_path: "D:\\Technical_works\\tensorflow\\own-data-embedding-visualization-vgg-16/embedding-logs\\metadata_4_classes.tsv"
            sprite {
        image_path: "D:\\Technical_works\\tensorflow\\own-data-embedding-visualization-vgg-16/embedding-logs\\sprite_4_classes.png"
        single_image_dim: 128
        single_image_dim: 128
        }
    }
    
    Change the metadata_path and image_path to your location where the metadata_4_classes.tsv
    and sprite_4_classes.png is located.
    
    
To run the embeddings launch tensor board 

     tensorboard --logdir=/path/to/your_log/embedding-logs --port=6006
     
     ## Please make sure there is no gap between the name of your directory-
        for e.g- folder name will not work it has to be folder_name
     
       Then open localhost:6006 in a browser
       
       Then go to the embedding options in Tensorboard
       
![Alt text]( https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization/blob/master/tensorboard.PNG?raw=true "tensorboard")

# To regenerate the embedding logs for feature vectors given in this repositoty

To regenerate the same embedding logs you can use the feature_vectors_400_samples.txt in the feature_vectors.zip file.

If you want to generate embedding visulaization for the given feature vector data, you can directly look into 
[own-data-embedding-visualization.py](https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization/blob/master/own-data-embedding-visualization.py) script to visualize your feature vectors in embedding visualizer.

The code is described block wise in the next section of # Generating the embedding logs for your own feature vectors
Running this script will generate the embedding logs specified to your system path .
Then you can run the tensorboard using

     tensorboard --logdir=/path/to/your log/embedding-logs --port=6006
     
       Then open localhost:6006 in a browser
       
       Then go to the embedding options in Tensorboard

# Data used in this Example
I have used 4 categories with 100 samples in each class - Cats, Dogs, Horses, Humans(Horse riders).The data are stored in data.zip folder
The Pretrained VGG16 is used to obtain feature vector of size 4096 from the penultimate layer of the network.

# Using VGG16 model to obtain feature vectors
If you want to use VGG16 as feature extractor for your own data you can look into [vgg16-feature-extraction.py](https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization/blob/master/vgg16-feature-extraction.py) script.
Download the VGG16 weights by reading the [VGG_model/download_vgg16_weights.md](https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization/blob/master/VGG_model/download_vgg16_weights.md)  and save it in VGG_model directory
The script will save your extracted features in feature_vectors.txt file as well as feature_vectors.pkl file. The shape of the obtained feature vector will be (num_samples,feature_vector_size).

    num_samples = number of images (in this example 400)
    feature_vector_size = size of feature vector for each image (in this example its 4096)

# Generating the embedding logs for your own feature vectors

If you want to generate embedding visulaization for your own feature vector data that you have- you can directly look into 
[own-data-embedding-visualization.py](https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization/blob/master/own-data-embedding-visualization.py) script to visualize your feature vectors in embedding visualizer.

Import the modules
      
      import os,cv2
      import numpy as np
      import matplotlib.pyplot as plt
      import pickle # if your feature vector is stored in pickle file
      import tensorflow as tf
      from tensorflow.contrib.tensorboard.plugins import projector

Define your log directory to store the logs

       PATH = os.getcwd()
       LOG_DIR = PATH+ '/embedding-logs'
       
Load the feature vectors and define the feature variable

       
       feature_vectors = np.loadtxt('feature_vectors_400_samples.txt')
       
       # the shape of feature vectors should be (num_samples,length_of_each_feature) . eg (400,4096)
       
       print ("feature_vectors_shape:",feature_vectors.shape) 
       print ("num of images:",feature_vectors.shape[0])
       print ("size of individual feature vector:",feature_vectors.shape[1])
       
       # Load the features in a tensorflow variable 
        features = tf.Variable(feature_vectors, name='features')
        
 Generate the metadeta files to assign class labels to the features:
 
       y = np.ones((num_of_samples,),dtype='int64')

       y[0:100]=0
       y[100:200]=1
       y[200:300]=2
       y[300:]=3

       names = ['cats','dogs','horses','humans']

       metadata_file = open(os.path.join(LOG_DIR, 'metadata_4_classes.tsv'), 'w')
       metadata_file.write('Class\tName\n')
       k=100 # number of samples in each class
       j=0

       for i in range(num_of_samples):
           c = names[y[i]]
           if i%k==0:
               j=j+1
           metadata_file.write('{}\t{}\n'.format(j,c))
           #metadata_file.write('%06d\t%s\n' % (j, c))
       metadata_file.close()
       
Load the image data that you want to visualize along with the label names on tensorboard

The shape of image data array should be (num_samples,rows,cols,channel) . In this example it is (400,224,224,3)

       img_data=[]
       for dataset in data_dir_list:
           img_list=os.listdir(data_path+'/'+ dataset)
           print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
           for img in img_list:
               input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
               input_img_resize=cv2.resize(input_img,(128,128)) # you can choose what size to resize your data
               img_data.append(input_img_resize)
       img_data = np.array(img_data)

Define the function to generate Sprite images. Sprite image is needed if you want to visualize the images along with
the label names for corresponding feature vectors.

       def images_to_sprite(data):
            """Creates the sprite image along with any necessary padding

            Args:
              data: NxHxW[x3] tensor containing the images.

            Returns:
              data: Properly shaped HxWx3 image with any necessary padding.
            """
            if len(data.shape) == 3:
                data = np.tile(data[...,np.newaxis], (1,1,1,3))
            data = data.astype(np.float32)
            min = np.min(data.reshape((data.shape[0], -1)), axis=1)
            data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
            max = np.max(data.reshape((data.shape[0], -1)), axis=1)
            data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
            # Inverting the colors seems to look better for MNIST
            #data = 1 - data

            n = int(np.ceil(np.sqrt(data.shape[0])))
            padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                    (0, 0)) + ((0, 0),) * (data.ndim - 3)
            data = np.pad(data, padding, mode='constant',
                    constant_values=0)
            # Tile the individual thumbnails into an image.
            data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                    + tuple(range(4, data.ndim + 1)))
            data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
            data = (data * 255).astype(np.uint8)
            return data
            
Generate the sprite image for your dataset

    sprite = images_to_sprite(img_data)
    cv2.imwrite(os.path.join(LOG_DIR, 'sprite_4_classes.png'), sprite)
    
For this example it looks like :

![Alt text](https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization/blob/master/embedding-logs/sprite_4_classes.png?raw=true)

Run a tensorflow session and write the log files in log directory

    with tf.Session() as sess:
        saver = tf.train.Saver([features])

        sess.run(features.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'images_4_classes.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_4_classes.tsv')
        # Comment out if you don't want sprites
        embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_4_classes.png')
        embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
        
The entire code is in [own-data-embedding-visualization.py](https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization/blob/master/own-data-embedding-visualization.py)
