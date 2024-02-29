#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "./matrix/Matrix.h"
#include "./network/NeuralNetwork.h"
#include "./image/Image.h"


#define LAYERS 3
#define BATCH_SIZE 1
#define TEST_SET_SIZE 200
#define TRAIN_SET_SIZE 1024

const int TOPOLOGY[LAYERS] = {784, 300, 10};
// const int TOPOLOGY[LAYERS] = {784, 300, 100, 50, 10};

double test_nn(NeuralNetwork* nn, char *path);
double train_nn(NeuralNetwork *nn, Batch** batches, int size, int epochs);
double calculateTime(struct timeval begin, struct timeval end);

int main(){
	Image **imgs_train = load_images("./data/mnist_train.csv", TRAIN_SET_SIZE);
	matrix **train_set = split_images(imgs_train, TRAIN_SET_SIZE);
	NeuralNetwork *nn = createNetwork(TOPOLOGY, LAYERS, 0.0095);
	// NeuralNetwork *nn = load_network("./parameters_v1", TOPOLOGY, LAYERS);

	struct timeval begin, end;

	//first test of the nn without training
	double accuracyFirst = test_nn(nn, "./data/mnist_test.csv");
	printf("Accuracy on test set (%d Images): %.2lf%% . \n", TEST_SET_SIZE, accuracyFirst);

	Batch** batches = split_into_mini_batches(train_set, TRAIN_SET_SIZE, BATCH_SIZE);

	gettimeofday(&begin, 0); //start training time

	double avgLoss = train_nn(nn, batches, TRAIN_SET_SIZE / BATCH_SIZE, 100);

	gettimeofday(&end, 0); // end training time

	double elapsed = calculateTime(begin , end);

	//test again after training
	double accuracy = test_nn(nn, "./data/mnist_test.csv");
	printf("Accuracy before training: %.2lf%% . \n", accuracyFirst);
	printf("Accuracy after training: %.2lf%% . \n", accuracy);
	printf("Average loss: %.2lf .\n", avgLoss);
	printf("Time measured: %.3f seconds.\n", elapsed);

	// save_network(nn, "./parameters_v3");

	freeNetwork(nn);
	return 0;
}

double train_nn(NeuralNetwork *nn, Batch** batches, int size, int epochs) {
	double loss = 0;
	for(int i = 0; i < epochs; i++){
		loss = 0;
		matrix *X = NULL, *Y = NULL, *Yhat;

		//mini-batch gradient descent
		for(int b = 0; b < size; b++){
			X = batches[b]->features, Y = batches[b]->labels;
			Yhat = forward(nn, X, true);
			loss += cross_entropy(Yhat, Y) / size;

			// back-propagation returns calculated gradients for each layer
			Gradient **grads = propagate(nn, Yhat, Y); 
			updateParameters(nn, grads);

			freeMatrix(Yhat);
			freeGradient(grads, nn->layers);
		}
		printf("Epoch[%d]: The loss >> %lf.\n", i + 1, loss);
	}

	return loss;
}

double test_nn(NeuralNetwork* nn, char *path) {
	Image **imgs_test = load_images(path, TEST_SET_SIZE);
	matrix **test_set = split_images(imgs_test, TEST_SET_SIZE);
	Batch** batches = split_into_mini_batches(test_set, TEST_SET_SIZE, BATCH_SIZE);
	int total = 0;
	for(int p = 0; p < TEST_SET_SIZE; p++){
		matrix *X = batches[p]->features, *Y = batches[p]->labels;
		matrix *Yhat = predict(nn, X, true);
		if(argmax(Yhat) == argmax(Y)) total += 1;
	}
	double percentage = (total * 100) / TEST_SET_SIZE;
	return percentage;
}

double calculateTime(struct timeval begin, struct timeval end){
	long seconds = end.tv_sec - begin.tv_sec;
	long microseconds = end.tv_usec - begin.tv_usec;
	return seconds + microseconds*1e-6;
}

