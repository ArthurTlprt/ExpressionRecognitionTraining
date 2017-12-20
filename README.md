# Expression recognition


If you want to use our model in your project, we suggest you to visit this repository.
This repository is dedicated to the preparation of the data and the training on [Affect-Net](https://arxiv.org/abs/1708.03985).

This is a deep learning model to classify facial expressions in five categories:
 * Neutral
 * Happy
 * Sad
 * Angry
 * Surprised

We implement different architectures:
 * [Inception-resnet](https://drive.google.com/file/d/0Bywjnd8SQSB6UmdEbzcyRF8xTWs/view)
 * Vgg16
 * [Inception-resnet v2](https://drive.google.com/file/d/0Bywjnd8SQSB6UmdEbzcyRF8xTWs/view)
 * [A 8 layers architecture](http://www.jeet.or.kr/LTKPSWeb/uploadfiles/be/201711/231120171455057903750.pdf)

To train and evaluate our networks, we used the database [Affect-Net](https://arxiv.org/abs/1708.03985), which is composed of 1M of images.

Our best accuracy is 74% on a validation set using Inception-resnet v2.

Here you can see a demo.
![Alt Text](https://github.com/ArthurTlprt/SentimentRecognition/blob/master/Demonstration.gif)
