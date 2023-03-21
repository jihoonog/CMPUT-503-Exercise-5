# CMPUT-503-Exercise-5
CMPUT 503 Exercise 5 ML on Duckiebot


# Deliverable 2 Questions and Answers

1. What data augmentation is used in training? Please delete the data augmentation and rerun the code to compare.

The image has been randomly rotated between positive and negative 5 degrees, in addition a random crop is added to the training image where a 2 pixel padding is applied to the image before taking a random 28x28 square crop essentially shifting the number within the image. Finally the data is then normalized in a PyTorch tensor. 
With the transformations the test accuracy was 97.91%. Without the transformation the test accuracy was 97.88%.

2. What is the batch size in the code? Please change the batch size to 16 and 1024 and explain the variation in results.

The default batch size is 64. With batch size 16 the test accuracy was 97.85% but training took much longer (twice as long) as with using a batch size of 64. With batch size 1024 the test accuracy was 97.45% and training still took a lot longer but still faster than training with a batch size of 16. The reason for the difference in training time could be caused by the overhead of transferring the mini-batches from the CPU's memory to the GPU's memory. A larger batch size will take longer to process but have less of them to process. While a smaller batch size will take less time but more of them to process. The overhead of transferring the images to the GPU's memory is multiplied by the number of batches, and this overhead scales with the number of images being transferred. Therefore, there is a happy medium where the training throughput is maximum while the total overhead of data being transferred is minimized. 

3. What activation function is used in the hidden layer? Please replace it with the linear activation function and see how the training output differs. Show your results before and after changing the activation function in your written report.

The default activation function is ReLU or Rectified Linear Unit. Using a linear activation function caused the test accuracy to drop down to 83.53%. In contrast using the ReLU activation function lead to a 98% accuracy for the test set. 

4. What is the optimization algorithm in the code? Explain the role of optimization algorithm in training process

The default optimization algorithm used is Adam. The optimization algorithm is used to update the parameters of the model with respect to the loss calculated during training.  

5. Add dropout in the training and explain how the dropout layer helps in training.

I added dropout layers after the 2 hidden layers after the non-linear activation function with a dropout probability of 0.1 or 10% and it produced a test accuracy of 97.58% on the test set. Dropout is useful when your model is overfitting on the training dataset. Because dropout randomly sets some of the nodes to 0, it effectively creates a new model where other nodes can learn from the training data.