# JLearn

## Description
JLearn is a machine learning library for Java. The library is built with easy setup and usage in mind, catering towards those who are interested in machine learning but don't know where to start. Currently, JLearn only supports Deep Learning networks, however, we plan to implement other algorithms such as Convolutional Neural Networks and Genetic Algorithms in the foreseeable future.

## Installation & Usage
- Download this repository and place the **JLearn** package into the source folder of your Java project

**Import Statement:**
```java
import JLearn.*;
```
- Then, to create a sequential model, use the `JLearn.Model` class
-  **Please Note:** The first and last Layers of the model must be the `Flattened` and `Classify` classes to perform proper predictions and Backpropagation

**Example:**
```java
Model model = new Model(new Layer[] { 
	new Flattened(784, 200, new int[] { -1, 1 }, new int[] { 0, 0 }),
	new Layer(200, 80, new int[] { -1, 1 }, new int[] { 0, 0 }),
	new Classify(80, 10, new int[] { -1, 1 }, new int[] { 0, 0 })
});
```
- Next, to train the model to fit a dataset, use the `model.train()` method

**Example:**
```java
model.train(3, 100, x_train , y_train, 0.01);
```
- Lastly, to make a prediction based off test input, use the `model.predict()` method

**Example:**
```java
model.predict(x_test[0]);
```

## Demonstration
- This is the cycle of a model using the **MNIST** dataset

**Training output:**
```
Epoch: 1, Duration: [171.967 seconds], Loss: 27322.11853781128
Epoch: 2, Duration: [169.053 seconds], Loss: 15735.0729208367
Epoch: 3, Duration: [168.714 seconds], Loss: 13177.855656711854
```
**Test input (image for clarity):**

![image](https://i.imgur.com/T3iRUl4.png)

**Test output (argMax for prediction, softmax for confidence):**
```
Prediction: 2, Confidence: 0.7206593891968534
```
## Credits
- Special thanks to [Jeff Griffith](https://github.com/jeffgriffith) for his [MnistReader](https://github.com/jeffgriffith/mnist-reader/blob/master/src/main/java/mnist/MnistReader.java) class

## License
- [MIT](https://github.com/shriramrav/JLearn/blob/master/LICENSE)
