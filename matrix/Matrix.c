#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Matrix.h"

matrix* createMatrix(unsigned int rows,unsigned int columns, int random){
	matrix* A = (matrix*)malloc(sizeof(matrix));
	A->rows = rows;
	A->columns = columns;
	A->entries = (double**)malloc(sizeof(double*) * rows);
	for(int i = 0; i < rows; i++){
		A->entries[i] = (double*)calloc(columns, sizeof(double));
	}
	if(random) randomize_matrix(A, 2);
	return A;
}

matrix* copyMatrix(matrix *A){
	if(A == NULL){
		printf("matrix is NULL, cannot copy.\n");
		return NULL;
	}
	matrix* B = createMatrix(A->rows, A->columns, false);
	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->columns; j++){
			B->entries[i][j] = A->entries[i][j];
		}
	}
	return B;
}

matrix* fill(matrix *A, double n){
	if(A == NULL) return A;
	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->columns; j++){
			A->entries[i][j] = n;
		}
	}
	return A;
}


void printMatrix(matrix* A){
	if(A == NULL) return;
	printf("[");
	for(int i = 0; i < A->rows; i++){
		printf("[");
		for(int j = 0; j < A->columns; j++){
			printf("%0.5lf", A->entries[i][j]);
			if(j != A->columns - 1) printf(", ");
		}
		printf("]");
		if(i != A->rows - 1) printf(",\n");
	}
	printf("]\n");
}


void freeMatrix(matrix* A){
	if(!A || A->rows == 0 || A->columns == 0) return;
	for(int i = 0; i < A->rows; i++){
		free(A->entries[i]);
	}
	free(A->entries);
	free(A);
	A = NULL;
}


/*========================================================== OPERATIONS ON MATRICES ==========================================================*/

matrix* add(matrix *A, matrix *B){
	if(A->rows != B->rows && B->columns != 1){
		printf("Dimension error, cannot add (%d, %d) to (%d, %d)", B->rows, B->columns, A->rows, A->columns);
		exit(1);
	}

	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->columns; j++){
			A->entries[i][j] += B->entries[i % B->rows][B->columns == 1 ? 0 : j];
		}
	}

	return A;
}

matrix* subtract(matrix *A, matrix *B){
	if(A->columns != B->columns || A->rows != B->rows){
		printf("Dimension error, cannot subtract (%d, %d) to (%d, %d)", A->rows, A->columns, B->rows, B->columns);
		exit(1);
	}

	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->columns; j++){
			A->entries[i][j] = A->entries[i][j] - B->entries[i][j];
		}
	}

	return A;
}

matrix* multiply(matrix *A, matrix *B){
	if(A->columns != B->rows){
		printf("Dimesions error, cannot multiply (%d, %d) x (%d, %d).\n", A->rows, A->columns, B->rows, B->columns);
		exit(1);
	}
	matrix *C = createMatrix(A->rows, B->columns, false);
	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < B->columns; j++){
			for(int k = 0; k < B->rows; k++){
				C->entries[i][j] += A->entries[i][k] * B->entries[k][j];
			}
		}
	}

	return C;
}

matrix* divide(matrix *A, matrix *B){
	if(A->columns != B->columns || A->rows != B->rows){
		printf("Dimesions error, cannot divide (%d, %d) x (%d, %d).\n", A->rows, A->columns, B->rows, B->columns);
		exit(1);
	}
	matrix *C = createMatrix(A->rows, B->columns, false);
	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < B->columns; j++){
			for(int k = 0; k < B->rows; k++){
				C->entries[i][j] += A->entries[i][k] / B->entries[k][j];
			}
		}
	}

	return C;
}

matrix* hadamard(matrix *A, matrix *B){
	if(A->columns != B->columns || A->rows != B->rows){
		printf("Dimension error, cannot multiply elementwise (%d, %d) to (%d, %d)", A->rows, A->columns, B->rows, B->columns);
		exit(1);
	}

	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->columns; j++){
			A->entries[i][j] = A->entries[i][j] * B->entries[i][j];
		}
	}

	return A;
}

double dot(matrix *A, matrix *B){
	if(A->rows > 1 || B->columns > 1 || A->columns != B->rows){
		printf("Dimensions error, cannot (%d, %d) dot (%d, %d).\n", A->rows, A->columns, B->rows, B->columns);
		exit(1);
	}

	double sum = 0;
	for(int i = 0; i < A->columns; i++)
		sum += A->entries[0][i] * B->entries[i][0];
	
	return sum;
}

matrix* scale(matrix *A, double n){
	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->columns; j++){
			A->entries[i][j] = A->entries[i][j] * n;
		}
	}

	return A;
}

matrix* T(matrix *A){
	matrix *B = createMatrix(A->columns, A->rows, false);
	for(int i = 0; i < B->rows; i++){
		for(int j = 0; j < B->columns; j++){
			B->entries[i][j] = A->entries[j][i];
		}
	}

	return B;
}

// (5, 2)
matrix* sum(matrix *A, int axis){ // axis = 1 (sum rows) axis = 0 (sum columns)
	int rows = axis ? A->rows : 1;
	int columns = axis ? 1 : A->columns;
	matrix *B = createMatrix(rows,columns, false); // 5, 1

	for(int i = 0; i < rows; i++){
		for(int j = 0; j < columns; j++){
			int elements = axis ? A->columns : A->rows;
			for(int k = 0; k < elements; k++){
				B->entries[i][j] += A->entries[axis ? i : k][axis ? k : i];
			}
		}
	}

	return B;
}

double summation(matrix *A){
	double sum = 0.0;
	for(int i = 0; i < A->rows; i++)
		for(int j = 0; j < A->columns; j++)
			sum += A->entries[i][j];
	return sum;
}

matrix* flatten(matrix *A){
	matrix *B = createMatrix(1, A->rows * A->columns, false);

	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->columns; j++){
			B->entries[0][j + A->columns * i] = A->entries[i][j];
		}
	}
	return B;
}

matrix* apply(matrix *A, double (*f)(double)){
	matrix *B = createMatrix(A->rows, A->columns, false);
	for(int i = 0; i < A->rows; i++)
		for(int j = 0; j < A->columns; j++)
			B->entries[i][j] = (*f)(A->entries[i][j]);

	return B;
}
/*========================================================== DETERMINANT ==========================================================*/
double det(matrix *A){
	if(A->rows != A->columns){
		printf("Error: determinants can only be calculated on sqaure matrices, dimensions of (n, n).\n");
		exit(1);
	}
	if(A->rows == 1 && A->columns == 1) return A->entries[0][0];
	if(A->rows == BASE && A->columns == BASE) return baseDet(A); // O(1)

	double sum = 0.0;
	for(int i = 0; i < A->rows; i++){
		matrix *M = minorA(A, i, 0); // O(1)
		sum += A->entries[i][0] * pow(-1, i) * det(M);
		freeMatrix(M); //O(1)
	}

	return sum;
}

static double baseDet(matrix *A){
	if(A->rows != BASE){
		printf("Error: only accepting (%d, %d) matrices.\n", BASE, BASE);
		exit(1);
	}
	return (A->entries[0][0] * A->entries[1][1]) - (A->entries[0][1] * A->entries[1][0]);
}

matrix *minorA(matrix *A, int row, int col){
	if(row >  A->rows - 1 || col > A->columns - 1) return NULL; 
	if(A->rows != A->columns){
		printf("Error: minors can only be calculated on sqaure matrices, (size of (n, n)).\n");
		exit(1);
	}

	int r = 0, c = 0; // row and column index for the created minor
	matrix *M = createMatrix(A->rows - 1, A->columns - 1, false);

	for(int i = 0; i < A->rows; i++){
		if (i == row) continue; // if the current row is being deleted then skip iteration

		c = 0;
		for(int j = 0; j < A->columns; j++){
			if (j != col){
				M->entries[r][c] = A->entries[i][j];
				c++;
			}
		}
		r++;
	}
	return M;
}

/*========================================================== PROBABILITY ==========================================================*/
matrix* softmax(matrix *A){
	double total = 0;
	matrix *B = createMatrix(A->rows, A->columns, false);
	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->columns; j++){
			total += exp(A->entries[i][j]);
		}
	}
	for(int i = 0; i < B->rows; i++){
		for(int j = 0; j < B->columns; j++){
			B->entries[i][j] = exp(A->entries[i][j]) / total;
		}
	}
	return B;
}

// expects a column vector
unsigned int argmax(matrix *A){ 
	int index = 0;
	double max = A->entries[0][0];
	for(int i = 0; i < A->rows; i++){
		if(A->entries[i][0] > max){
			max = A->entries[i][0];
			index = i;
		}
	}
	return index;
}

/*========================================================== RANDOM ==========================================================*/

double uniform_distribution(double low, double high) {
	double difference = high - low; // The difference between the two
	int scale = 10000;
	int scaled_difference = (int)(difference * scale);
	return low + (1.0 * (rand() % scaled_difference) / scale);
}

void randomize_matrix(matrix* m, int n) {
	double min = -1.0 / sqrt(n);
	double max = 1.0 / sqrt(n);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->columns; j++) {
			m->entries[i][j] = uniform_distribution(min, max);
		}
	}
}
/*========================================================== LOAD & SAVE TO FILE ==========================================================*/
// Function to save a matrix to a file
void save_matrix(const matrix *A, const char *fileName) {
	FILE *file = fopen(fileName, "w");
	if (file == NULL) {
		perror("Error opening file");
		exit(EXIT_FAILURE);
	}

	fprintf(file, "%d %d\n", A->rows, A->columns);

	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < A->columns; j++) {
			fprintf(file, "%lf ", A->entries[i][j]);
		}
		fprintf(file, "\n");
	}

	fclose(file);
}

// Function to load a matrix from a file
matrix* load_matrix(const char *fileName) {
	FILE *file = fopen(fileName, "r");
	if (file == NULL) {
		perror("Error opening file");
		exit(EXIT_FAILURE);
	}

	matrix *A;
	int rows, cols;
	fscanf(file, "%d %d", &rows, &cols);

	A = createMatrix(rows, cols, false);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			fscanf(file, "%lf", &(A->entries[i][j]));
		}
	}

	fclose(file);

	return A;
}
