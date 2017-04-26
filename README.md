# Tensorboard-own-image-data-image-features-embedding-visualization
Learn how to visualize your own image data or features on Tensorboard Embedding Visualizer

To run the embeddings already provided in embedding-logs. Download all the files.

    In the embedding-logs/projector_config.pbtxt 
       
            embeddings {
         tensor_name: "features:0"
         metadata_path: "D:\\Technical_works\\tensorflow\\own-data-embedding-visualization-vgg-16/embedding-            logs\\metadata_4_classes.tsv"
            sprite {
        image_path: "D:\\Technical_works\\tensorflow\\own-data-embedding-visualization-vgg-16/embedding-logs\\sprite_4_classes.png"
        single_image_dim: 224
        single_image_dim: 224
        }
    }
    
    
    Change the metadata_path and image_path to your location where the metadata_4_classes.tsv and sprite_4_classes.png is located.
    
    
To run the embeddings launch tensor board 

     tensorboard --logdir=/path/to/your log/embedding-logs --port=6006
     
       Then open localhost:6006 in a browser
       
       Then go to the embedding options in Tensorboard
       

I have used 4 categories with 100 samples in each class - Cats, Dogs, Horses, Humans(Horse riders).
The Pretrained VGG16 is used to obtain feature vector of size 4096 from the penultimate layer of the network.

If you want to use VGG16 as feature extractor for your own data you can look into vgg16-feature-extraction.py script.
The script will save your extracted features in feature_vectors.txt file as well as feature_vectors.pkl file. The shape of the obtained feature vector will be (num_samples,feature_vector_size).

    num_samples = number of images (in this example 400)
    feature_vector_size = size of feature vector for each image (in this example its 4096)
    
If you already have your feature vector data or you used some other network to obtain your feature vectors, you can directly look into 
own-data-embedding-visualization.py script to visualize your feature vectors in embedding visualizer.
