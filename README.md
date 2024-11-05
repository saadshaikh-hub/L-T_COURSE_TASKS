# Deep Learning Project Tasks

This repository contains code implementations and reports for a series of tasks related to Deep Learning, covering a range of techniques and architectures. Each task is designed to enhance understanding of different aspects of neural networks, regularization, image classification, augmentation, transfer learning, and generative models.

## Table of Contents

- [Task 1: Exploring Activation Functions in Deep Learning](#task-1-exploring-activation-functions-in-deep-learning)
- [Task 2: Regularization Techniques for Deep Learning Models](#task-2-regularization-techniques-for-deep-learning-models)
- [Task 3: Building and Evaluating a CNN for Multi-class Image Classification](#task-3-building-and-evaluating-a-cnn-for-multi-class-image-classification)
- [Task 4: Impact of Image Augmentation on CNN Performance](#task-4-impact-of-image-augmentation-on-cnn-performance)
- [Task 5: Leveraging Transfer Learning for Image Classification](#task-5-leveraging-transfer-learning-for-image-classification)
- [Task 6: Machine Translation with Sequence-to-Sequence Learning](#task-6-machine-translation-with-sequence-to-sequence-learning)
- [Task 7: Image Denoising with Denoising Autoencoders](#task-7-image-denoising-with-denoising-autoencoders)
- [Task 8: Exploring Generative Image Creation with DCGANs](#task-8-exploring-generative-image-creation-with-dcgans)

---

### Task 1: Exploring Activation Functions in Deep Learning

**Objective:**  
Implement and evaluate different activation functions (Sigmoid, Tanh, ReLU) within a neural network on the MNIST dataset.

**Requirements:**  
- Simple neural network with one hidden layer.
- Implement Sigmoid, Tanh, and ReLU activation functions.
- Train and evaluate accuracy for each activation function.
- Analyze the behavior and performance impact of each function.

**Deliverables:**  
1. Python code implementing the neural network with all activation functions.
2. Comparison chart of accuracy for each activation function.
3. Report explaining observations and recommending the best-performing function.

---

### Task 2: Regularization Techniques for Deep Learning Models

**Objective:**  
Use Ridge and Lasso regularization on a CNN for CIFAR-10 image classification to analyze the effects on overfitting and underfitting.

**Requirements:**  
- Implement a CNN in TensorFlow/PyTorch.
- Apply various strengths of Ridge and Lasso regularization.
- Compare accuracy and loss on training and validation sets.
- Determine optimal regularization strength.

**Deliverables:**  
1. Python code implementing CNN with Ridge and Lasso.
2. Plots showing regularization impact on loss and accuracy.
3. Report explaining overfitting/underfitting observations and the chosen strength.

---

### Task 3: Building and Evaluating a CNN for Multi-class Image Classification

**Objective:**  
Design and implement a CNN for multi-class classification using a dataset with multiple categories, such as CIFAR-100.

**Requirements:**  
- CNN with convolutional and pooling layers.
- Preprocess images: resizing, normalization, and one-hot encoding.
- Train and evaluate using accuracy, precision, recall, and F1-score.
- Visualize performance with confusion matrices.

**Deliverables:**  
1. Python code for the CNN architecture.
2. Report detailing architecture and hyperparameter choices.
3. Evaluation metrics and confusion matrix for each class.

---

### Task 4: Impact of Image Augmentation on CNN Performance

**Objective:**  
Compare CNN performance with and without image augmentation on a classification task.

**Requirements:**  
- Implement a CNN and train it on the dataset without augmentation.
- Apply augmentation techniques (cropping, flipping, rotation) in a separate model.
- Compare accuracy and loss for both models on the test set.

**Deliverables:**  
1. CNN code with augmentation techniques.
2. Table/chart showing performance with and without augmentation.
3. Report on how augmentation affects overfitting and performance.

---

### Task 5: Leveraging Transfer Learning for Image Classification

**Objective:**  
Use transfer learning by fine-tuning a pre-trained CNN (e.g., VGG16 or ResNet) on a smaller dataset with limited classes.

**Requirements:**  
- Choose a pre-trained model and freeze initial layers.
- Fine-tune final layers on a new dataset.
- Train and compare with a model trained from scratch.

**Deliverables:**  
1. Fine-tuning code for the pre-trained CNN.
2. Comparison table of test accuracy and loss.
3. Report explaining transfer learning benefits in limited data settings.

---

### Task 6: Machine Translation with Sequence-to-Sequence Learning

**Objective:**  
Develop a Seq2Seq model with LSTMs for translating between two languages (e.g., English to French).

**Requirements:**  
- Implement Seq2Seq with LSTMs as encoders and decoders.
- Train on parallel sentence data.
- Evaluate translation accuracy with BLEU or ROUGE scores.

**Deliverables:**  
1. Seq2Seq model code.
2. Report describing model architecture and training.
3. Evaluation results with BLEU or ROUGE scores.

---

### Task 7: Image Denoising with Denoising Autoencoders

**Objective:**  
Build a Denoising Autoencoder to clean noisy images.

**Requirements:**  
- Implement an autoencoder with encoding and decoding layers.
- Train on noisy-clean image pairs.
- Visually and quantitatively compare denoised images.

**Deliverables:**  
1. Denoising Autoencoder code.
2. Visual comparisons between noisy and denoised images.
3. Report on PSNR and effectiveness of the autoencoder.

---

### Task 8: Exploring Generative Image Creation with DCGANs

**Objective:**  
Generate realistic images using a DCGAN trained on a dataset (e.g., faces or landscapes).

**Requirements:**  
- Implement a DCGAN with Generator and Discriminator.
- Train and generate new images.
- Visually assess realism and adherence to dataset features.

**Deliverables:**  
1. DCGAN code for image generation.
2. Visualization of generated vs. real images.
3. Report on image quality, challenges, and potential improvements.

---

Each task is implemented in Python, using TensorFlow or PyTorch where required, with additional explanations provided in individual reports to assist in understanding the outcomes and justifications.
```
