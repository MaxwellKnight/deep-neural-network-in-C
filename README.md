# Neural Network Project README

This project was developed with the aim of implementing a neural network for image classification using the MNIST dataset. Throughout the development process, one of the key challenges revolved around ensuring the correctness of both forward and backward propagation. Achieving accurate dimensions for each neuron and layer during forward propagation was essential for the proper functioning of the neural network. Similarly, implementing backpropagation and accurately calculating gradients posed significant challenges.

### Model Training Results

The culmination of this project led to the successful training of a simple neural network model on the MNIST dataset. The model achieved an impressive 93% accuracy on the test set, demonstrating the effectiveness of the implemented neural network and the underlying matrix library.

## Project Overview

The neural network project is divided into two primary components: the neural network module and a matrix library that supports various operations on matrices.

## Neural Network Module

### `NeuralNetwork` Struct

```cpp
struct NeuralNetwork {
	int *topology;
	int layers;
	double learning_rate;
	matrix **W;
	matrix **b;
	Cache **caches;
};
```

NeuralNetwork structure encapsulates the neural network model, including the number of layers, learning rate, weight matrices, bias matrices, and caches for intermediate values during forward propagation.

### Neural Network Initialization (`createNetwork`)

```cpp
NeuralNetwork* createNetwork(const int topology[], int n, double learning_rate);
```

Description: Allocates memory for the neural network and initializes its parameters.
Parameters:
`const int *topology`: Array specifying the number of neurons in each layer.
`int n`: Number of layers.
`double learning_rate`: Learning rate for gradient descent.
Returns: A pointer to the initialized NeuralNetwork structure.

### Forward Propagation (`forward`)

```cpp
matrix* forward(NeuralNetwork *nn, matrix *X, int isMulticalss);
```

Description: Performs forward propagation through the neural network.
Parameters:
`NeuralNetwork *nn`: Pointer to the neural network.
`matrix *X`: Input matrix.
`int isMulticlass`: Flag for multi-class classification.
Returns: Output matrix after forward propagation.

### Cost Functions (`cost` and `cross_entropy`)

```cpp
double cross_entropy(matrix* AL, matrix* Y);
double cost(matrix* AL, matrix* Y);
```

Description: Computes the cost function for binary and multi-class classification.
Parameters:
`matrix* AL`: Output matrix after forward propagation.
`matrix* Y`: True labels matrix.
Returns: Cost value.

### Prediction Function (`predict`)

```cpp
matrix* predict(NeuralNetwork *nn, matrix* X, int isMulticlass);
```

Description: Performs forward propagation and modifies the output for multi-class classification.
Parameters:
`NeuralNetwork *nn`: Pointer to the neural network.
`matrix* X`: Input matrix.
int isMulticlass: Flag for multi-class classification.
Returns: Modified output matrix for multi-class classification.

### Backward Propagation (`propagate`)

```cpp
Gradient** propagate(NeuralNetwork *nn, matrix *AL, matrix *Y);
```

Description: Computes gradients for weights and biases using backward propagation.
Parameters:
`NeuralNetwork *nn`: Pointer to the neural network.
`matrix *AL`: Output matrix after forward propagation.
`matrix *Y`: True labels matrix.
Returns: Array of gradients for each layer.

### Update Parameters (`updateParameters`)

```cpp
void updateParameters(NeuralNetwork *nn, Gradient **grads);

```

Description: Updates weights and biases using gradient descent.
Parameters:
`NeuralNetwork *nn`: Pointer to the neural network.
`Gradient **grads`: Array of gradients for each layer.

### Activation Functions (`sigmoid`, `relu`, `sigmoid_backward`, `relu_backward`)

```cpp
double sigmoid_backward(double x);
double sigmoid(double x);
double relu(double x);
double relu_backward(double x);

```

Description: Implements sigmoid and ReLU activation functions and their corresponding backward derivatives.

## Matrix Library

```cpp
# Operations on matrices
matrix* add(matrix *A, matrix *B);
matrix* subtract(matrix *A, matrix *B);
matrix* multiply(matrix *A, matrix *B);
matrix* divide(matrix *A, matrix *B);
matrix* hadamard(matrix *A, matrix *B);
matrix* scale(matrix *A, double n);
matrix* sum(matrix *A, int axis); // axis = 1 (sum rows) axis = 0 (sum columns)
double summation(matrix *A);
matrix* flatten(matrix *A);
matrix* apply(matrix *A, double (*f)(double));
double dot(matrix *A, matrix *B);
matrix* T(matrix *A);

# determinant
static double baseDet(matrix *A);
matrix* minorA(matrix *matrix, int row, int col);
double det(matrix *A);

# Probability
matrix* softmax(matrix *A);
unsigned int argmax(matrix *A); // A is a column vector

```

The matrix library provides basic operations on matrices, including addition, subtraction, multiplication, division, element-wise operations, and transpose.

## Matrix Library Usage

Matrix Initialization:

Use `createMatrix` to initialize matrices.
Use `copyMatrix` to create a deep copy of a matrix.
Use `fill` to fill a matrix with a specified value.

### Matrix Operations:

Use operations such as `add`, `subtract`, `multiply`, etc., to perform element-wise operations.\
Use `hadamard` for element-wise multiplication.\
Use `scale` to scale matrix entries by a constant.\
Use `sum` and summation for summation along specified axes.\
Use `flatten` to convert a matrix into a vector.\
Use `apply` to apply a function element-wise to a matrix.\
Use `dot` for the dot product of matrices.\
Use `T` to transpose a matrix.\

### Determinant Functions:

Use `det` to compute the determinant of a matrix.\

### Probability Functions:

Use `softmax` for applying the softmax function.\
Use `argmax` to find the index of the maximum value in a column vector.\

### Random Functions:

Use `uniform_distribution` to generate random numbers.\
Use `randomize_matrix` to randomize the entries of a matrix.\

### Load and Save Functions:

Use `load_matrix` to load a matrix from a file.\
Use `save_matrix` to save a matrix to a file.\

## Contribution and License

Feel free to contribute to this project by opening issues or pull requests.
