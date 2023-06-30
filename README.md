# cGAN-CIFAR-10
## Abstract:
This blog post provides an overview of the knowledge and skills I acquired during my personal project focused on the CIFAR-10 dataset, specifically studying Generative Adversarial Networks (GANs), model parameters, and hyperparameters. I will highlight the implementation of GANs on the CIFAR-10 dataset, along with my exploration of hyperparameter tuning for improved results.

During this project, I delved into the fundamentals of GANs, a powerful deep learning model used for generating synthetic data. I learned about their significance in various applications, including image generation and data augmentation, specifically tailored to the CIFAR-10 dataset.

I will discuss the practical implementation of GANs on the CIFAR-10 dataset, where I followed a step-by-step process to generate realistic images. By experimenting with different model architectures and training techniques, I was able to improve the quality of the generated images and ensure they align with the unique characteristics of the CIFAR-10 dataset.

Furthermore, I explored the importance of model parameters and hyperparameters in the training process. I learned how these factors affect the performance and quality of the generated images specifically for the CIFAR-10 dataset. Through meticulous hyperparameter tuning, I made adjustments to enhance the performance of the GAN models and generate even more compelling and realistic images.

As future work, I propose leveraging the skills I acquired to further enhance the generation of synthetic images on the CIFAR-10 dataset. By fine-tuning the model parameters and hyperparameters, I aim to push the boundaries of image generation and explore novel applications such as data augmentation for CIFAR-10-based tasks.

This blog post showcases my personal project journey, highlighting the practical implementation of GANs on the CIFAR-10 dataset, discussing the impact of model parameters and hyperparameters specific to CIFAR-10, and outlining the potential future applications of these skills in image generation and data augmentation.

## Implementation of cGAN on CIFAR-10 Dataset:

> **Description of the CIFAR-10 Dataset:** The CIFAR-10 dataset is a widely used benchmark dataset in the field of computer vision. It consists of 60,000 color images in 10 different classes, with each class containing 6,000 images. The dataset is split into 50,000 training images and 10,000 testing images. The images in CIFAR-10 are relatively small, with a size of 32x32 pixels, and are categorized into the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is balanced, meaning each class has an equal number of images. Due to its widespread use and availability, the CIFAR-10 dataset has become a standard benchmark for developing and comparing various image classification algorithms and techniques.

**Dataset Preparation:** The first step in the implementation process was to prepare the CIFAR-10 dataset. The CIFAR-10 dataset consists of 50,000 training images and 10,000 test images. Each image is a color image of size 32x32 pixels. I used the Tensorflow framework to load and preprocess the dataset.

**Building the Generator Network:** The generator network is responsible for generating fake images that resemble the real images from the dataset. I implemented a deep convolutional neural network (CNN) as the generator network. The network takes a random noise vector along with a label (condition) as input and generates an image of size 32x32 pixels. The final architecture is as shown below.
![Demo Link](https://github.com/sujay-2001/cGAN-CIFAR-10/blob/main/cifar_gen.png)

**Building the Discriminator Network:** The discriminator network is responsible for distinguishing between real and fake images. It is also implemented as a deep CNN. The discriminator network takes an image along with a label (condition) as input and outputs a probability indicating whether the image is real or fake.
![Demo Link](https://github.com/sujay-2001/cGAN-CIFAR-10/blob/main/cifar_dis.png)
  
**Training the GAN:** The GAN training process involves training the generator and discriminator networks in an adversarial manner. The training process I employed involves alternating between training the discriminator on real and fake images and training the generator to improve its ability to generate realistic images. The loss function used is Binary Cross Entropy and optimiser is Adam. Overall the parameters of Generator and Discriminator are updated based on the following equation:

**Evaluation and Tuning the Hyper-parameters of the model:** After training the GAN on the CIFAR-10 dataset, I evaluated the performance of the generator network by generating a set of fake images for each class and comparing them. I also observed how well the model is able to learn the condition, and how it performs with respect to controlled generation. I also calculated metrics such as accuracy and loss to measure the performance of the discriminator and generator networks. I used the above metrics to tune the hyperparameters of the model. This is one of the most challenging task I had to work on.
Hyperparameters:
> Kernel size (=5(Discriminator),7(Generator): Kernel sizes of (5,5) in Discriminator network, (7,7) in Generator network gave best results among other choices. Any size greater than this leads to increased complexity, and hence slow learning.

> No of layers (=4): The no of layers in the CNN is fixed to be 4, since it is the ideal choice where any values greater than this leads to increased complexity, slower training, and lower than this leads to unsatisfactory results.
> Learning Rate (=0.0001): lr = 0.001 led to unstable training (generator unable to learn with discriminator), while lr = 0.00001 led to slower training, After careful consideration, lr = 0.0001 was chosen.

> Batch Size (=128): Choosing batch size of 128 showed stable learning, while any sizes greater or lesser led to unstable training. Another important observation is that batch size has significant correlation with learning rate with respect to performance.

> Choice of Activation functions: Relu in Generator network and LeakyRelu with α = 0.01 in Discriminator network gave best results as shown.

> Dropout =0.4: Addition of dropouts in final flattened layer of discriminator network improved results.
> No of Epochs =100: Results are good after 100 epochs with 128 as batch size.
> 
**Results:** The quality of the generated images is satisfactory and distinguishable. The final accuracy of the model is evaluated by computing the mean square error (MSE) of the normalized pixel values of generated image and real images for each class. The evaluated accuracy is 82%.

## Extension ideas:
Effect of changing latent space.

Effect of introducing skip connections.

Effect of adding L2/SSIM loss.
