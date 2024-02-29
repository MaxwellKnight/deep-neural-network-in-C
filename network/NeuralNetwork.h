#ifndef _NEURAL_NET
#define _NEURAL_NET
#include "../matrix/Matrix.h"

typedef struct NeuralNetwork NeuralNetwork;
typedef struct Cache Cache;
typedef struct Gradient Gradient;

struct NeuralNetwork {
	int *topology;
	int layers;
	double learning_rate;
	matrix **W;
	matrix **b;
	Cache **caches;
};

struct Gradient {
	matrix *dW;
	matrix *db;
	matrix *dZ;
};

/* cache structure to store intermediate calculations in order to calculate derivatives */
struct Cache {
	matrix *A_prev;
	matrix *Z;
};


NeuralNetwork* createNetwork(const int topology[], int n, double learning_rate);

/* FORWARD PASS */
matrix* forward(NeuralNetwork *nn, matrix *X, int isMulticalss);
Cache *cache(matrix *A_prev, matrix *Z);
double cost(matrix* AL, matrix* Y);
double cross_entropy(matrix* AL, matrix* Y);

matrix* predict(NeuralNetwork *nn, matrix* X, int isMulticlass);

/* BACK PROPAGATION */
Gradient** propagate(NeuralNetwork *nn, matrix *AL, matrix *Y);
void updateParameters(NeuralNetwork *nn, Gradient **grads);
Gradient* createGradient(matrix *dW, matrix *db, matrix *dZ);

/* MEMORY MANAGEMENT */
void freeCache(Cache **cache, int L);
void freeGradient(Gradient **grads, int L);
void freeNetwork(NeuralNetwork *nn);

void save_network(const NeuralNetwork *nn, const char *folderPath);
NeuralNetwork* load_network(const char *folderPath, const int *topology, int n);
void printNetwork(NeuralNetwork *nn);

/* ACTIVATIONS */
double sigmoid_backward(double x);
double sigmoid(double x);
double relu(double x);
double relu_backward(double x);

#endif