#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include "./NeuralNetwork.h"

NeuralNetwork* createNetwork(const int *topology, int n, double learning_rate){
	NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

	nn->layers = n - 1;
	nn->learning_rate = learning_rate;
	nn->caches = NULL;
	nn->W = (matrix**)calloc(n - 1, sizeof(matrix*));
	nn->b = (matrix**)calloc(n - 1, sizeof(matrix*));
	nn->caches = NULL;

	for(int l = 0; l < nn->layers; l++){
        nn->W[l] = createMatrix(topology[l + 1], topology[l], true);
        nn->b[l] = createMatrix(topology[l + 1], 1, false);
    }
	
	return nn;
}

Cache *cache(matrix *A_prev, matrix *Z){
	Cache *cache = (Cache*)malloc(sizeof(Cache));
	cache->A_prev = copyMatrix(A_prev);
	cache->Z = copyMatrix(Z);
	return cache;
}

Gradient* createGradient(matrix *dW, matrix *db, matrix *dZ){
	Gradient *grad = (Gradient*)malloc(sizeof(Gradient));
	grad->dW = copyMatrix(dW);
	grad->db = copyMatrix(db);
	grad->dZ = copyMatrix(dZ);
	return grad;
}
/*========================================================== FORWARD PROPAGATION ==========================================================*/
matrix* forward(NeuralNetwork *nn, matrix *X, int isMulticalss){
	int L = nn->layers - 1;
	matrix *A = X, *Z = NULL, *W = NULL, *b = NULL, *AL = NULL;
	nn->caches = (Cache**)malloc(sizeof(Cache*) * L);

	for(int l = 0; l < L; l++){
		matrix *A_prev = A;
		W = nn->W[l], b = nn->b[l];

		Z = add(multiply(W, A_prev), b); // Z[l] = W[l] * A[l - 1] + b[l]
		nn->caches[l] = cache(A_prev, Z); // caching  (for all layers l <= L - 1)
		A = apply(Z, sigmoid); // A[l] = sigmoid(Z[l])
	}

	W = nn->W[L], b = nn->b[L];

	Z = add(multiply(W, A), b); // Z[L] = W[L] * A[L - 1] + b[L]
	nn->caches[L] = cache(A, Z); //caching for last layer (layer l = L)

	AL = isMulticalss ? softmax(Z) : apply(AL, sigmoid); 

	return AL;
}

// Function to compute the cost using the given vectors AL (Yhat) and Y for multi-class classification
double cross_entropy(matrix* AL, matrix* Y) {
	double m = (double)Y->columns; // # of labels

	matrix* AL_dot_Y = hadamard(apply(AL, log), Y); // Y_i * log(AL_i)
	double L = summation(AL_dot_Y); // ∑[Y_i * log(AL_i)]

	// free intermediate calculations
	freeMatrix(AL_dot_Y);
	return (-1.0 / m) * L;
}

//function to predict a class for a single input 
matrix* predict(NeuralNetwork *nn, matrix* X, int isMulticlass){
	matrix *Ypred = forward(nn, X, isMulticlass);
	if(isMulticlass){
		int predIndex = argmax(Ypred);
		fill(Ypred, 0)->entries[predIndex][0] = 1;
		return Ypred;
	}
	return Ypred;
}

/*========================================================== BACKWARD PROPAGATION ==========================================================*/
Gradient** propagate(NeuralNetwork *nn, matrix *AL, matrix *Y){
	int L = nn->layers - 1;
	double m = (double)AL->columns;
	Gradient **grads = (Gradient**)malloc(sizeof(Gradient*) * nn->layers);
	matrix *A_prev = T(nn->caches[L]->A_prev);


	matrix *dZ = subtract(copyMatrix(AL), Y); // dZ[L] = AL - Y
	matrix *dW = scale(multiply(dZ, A_prev), 1.0 / m); // dW[L] = (1.0 / m) * dZ[L] * AL[L]
	matrix *db = scale(sum(dZ, 1), 1.0 / m);// ∑dZ_i
	grads[L] = createGradient(dW, db, dZ);

	//free intermediate calculations
	freeMatrix(dW); freeMatrix(db); freeMatrix(dZ); freeMatrix(A_prev);

	for(int l = L - 1; l >= 0; l--){
		Cache* current_cache = nn->caches[l];
		matrix *Z_relu = apply(current_cache->Z, sigmoid_backward);
		matrix *W_next = T(nn->W[l + 1]), *A = T(current_cache->A_prev);

		dZ = hadamard(multiply(W_next, grads[l + 1]->dZ), Z_relu); // dZ[l] = W[l+1] * dZ[l+1] (elementwise) g'(Z)[l]
		dW = scale(multiply(dZ, A), 1.0 / m); // dW[l] = dZ[l] * A[l]
		db = scale((sum(dZ, 1)), 1.0 / m);
		grads[l] = createGradient(dW, db, dZ);
		
		//free intermediate calculations
		freeMatrix(dZ); freeMatrix(dW); freeMatrix(db);
		freeMatrix(A); freeMatrix(Z_relu); freeMatrix(W_next);
	}

	freeCache(nn->caches, nn->layers);
	return grads;
}	

void updateParameters(NeuralNetwork *nn, Gradient **grads) {
	for (int l = 0; l < nn->layers; l++) {
		nn->W[l] = subtract(nn->W[l], scale(grads[l]->dW, nn->learning_rate)); // W[l] = W[l] - alpha * dW[l]
		nn->b[l] = subtract(nn->b[l], scale(grads[l]->db, nn->learning_rate)); // b[l] = b[l] - alpha * db[l]
	}
}
/*========================================================== ACTIVATIONS ==========================================================*/

double sigmoid(double x){ return 1 / (1 + exp(-x)); }
double relu(double x){ return x > 0 ? x : 0; }
double relu_backward(double x){ return x <= 0 ? 0 : x; }
double sigmoid_backward(double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }


void printNetwork(NeuralNetwork *nn){
	for(int l = 0; l < nn->layers; l++){
		matrix *A = nn->W[l];
		matrix *b = nn->b[l];
		printf("L = %d | Weights shape:(%d, %d)\n",l + 1, A->rows, A->columns);
		printMatrix(A);
		printf("\nL = %d | Bias shape:(%d, %d)\n", l + 1, b->rows, b->columns);
		printMatrix(nn->b[l]);
		printf("\n");
		if(l < nn->layers - 1) printf("\n");
	}
}

/*========================================================== LOAD & SAVE TO/FROM A FILE ==========================================================*/
// Function to save the parameters of the neural network to a file
void save_network(const NeuralNetwork *nn, const char *folderPath) {
	// Create a new folder inside the current folder
	char folder[256];
	sprintf(folder, "%s", folderPath);
	mkdir(folder, 0777); 
	
	// Save parameters to individual files
	for (int l = 0; l < nn->layers; l++) {
		char fileName[256];
		sprintf(fileName, "%s/layer%d_W.txt", folder, l + 1);
		save_matrix(nn->W[l], fileName);

		sprintf(fileName, "%s/layer%d_b.txt", folder, l + 1);
		save_matrix(nn->b[l], fileName);
	}

	// Save learning rate
	char lrFileName[256];
	sprintf(lrFileName, "%s/learning_rate.txt", folder);
	FILE *lrFile = fopen(lrFileName, "w");
	fprintf(lrFile, "%lf", nn->learning_rate);
	fclose(lrFile);
}

// Function to load the parameters of the neural network from a given folder
NeuralNetwork* load_network(const char *folderPath, const int *topology, int n) {
    NeuralNetwork *nn = createNetwork(topology, n, 0.0); 

    // Load parameters from individual files
    for (int l = 0; l < nn->layers; l++) {
        char fileName[256];
        sprintf(fileName, "%s/layer%d_W.txt", folderPath, l + 1);
        nn->W[l] = load_matrix(fileName);

        sprintf(fileName, "%s/layer%d_b.txt", folderPath, l + 1);
        nn->b[l] = load_matrix(fileName);
    }

    // Load learning rate
    char lrFileName[256];
    sprintf(lrFileName, "%s/learning_rate.txt", folderPath);
    FILE *lrFile = fopen(lrFileName, "r");
    fscanf(lrFile, "%lf", &(nn->learning_rate));
    fclose(lrFile);

    return nn;
}

/*========================================================== FREE MEMORY ==========================================================*/
void freeGradient(Gradient **grads, int L){
	for(int l = 0; l < L - 1; l++){
		freeMatrix(grads[l]->dW);
		freeMatrix(grads[l]->db);
		freeMatrix(grads[l]->dZ);
	}
	free(grads);
	grads = NULL;
}

void freeCache(Cache **caches, int L){
	for (int l = 0; l < L; l++){
		freeMatrix(caches[l]->A_prev);
		freeMatrix(caches[l]->Z);
	}
	free(caches);
}

void freeNetwork(NeuralNetwork *nn) {
	for(int l = 0; l < nn->layers; l++){
		freeMatrix(nn->W[l]);
		freeMatrix(nn->b[l]);
	}
	free(nn->W);
	free(nn->b);
	free(nn);
	nn = NULL;
}
