# Expression recognition

This is a deep learning model to classify facial expressions in five categories:
 * Neutral
 * Happy
 * Sad
 * Angry
 * Surprised

We implement different architectures:
 * Inception-resnet
 * Vgg16
 * Inception-resnet v2
 * A 8 layers architecture

To train and evaluate our networks, we used the database Affect-Net, which is composed of 1M of images.

Our best accuracy is 74% on a validation set using Inception-resnet v2.

Here you can see a demo.
![Alt Text](https://github.com/ArthurTlprt/SentimentRecognition/blob/master/Demonstration.gif)
