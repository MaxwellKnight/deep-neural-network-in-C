#ifndef _MATRIX
#define _MATRIX
#define BASE 2

typedef struct Matrix matrix;
typedef enum { false, true } BOOL;

struct Matrix {
	double **entries;
	unsigned int rows;
	unsigned int columns;
};

matrix* createMatrix(unsigned int rows, unsigned int columns, int random);
matrix* copyMatrix(matrix *A);
matrix* fill(matrix* A, double n);
void freeMatrix(matrix* A);
void printMatrix(matrix* A);

/* Operations on matrices */
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

/* determinant */
static double baseDet(matrix *A);
matrix* minorA(matrix *matrix, int row, int col);
double det(matrix *A);

/* Probability */
matrix* softmax(matrix *A);
unsigned int argmax(matrix *A); // A is a column vector

matrix* load_matrix(const char *fileName);
void save_matrix(const matrix *mat, const char *fileName);

/* random */
double uniform_distribution(double low, double high);
void randomize_matrix(matrix* m, int n);
#endif
