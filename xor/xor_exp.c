#include "./xor_exp.h"

matrix* XOR_X(){
	matrix *X = createMatrix(2,4, false);
	double arr[2][4] = {{0.0, 1.0, 0.0, 1.0},
							  {0.0, 0.0, 1.0, 1.0}};
	for(int i = 0; i < X->rows; i++){
		for(int j = 0; j < X->columns; j++)
			X->entries[i][j] = arr[i][j];
	}
	return X;
}
matrix* XOR_Y(){
	matrix *Y = createMatrix(1,4, false);
	double arr[1][4] = {{0.0, 1.0, 1.0, 0.0}};
	for(int i = 0; i < Y->rows; i++){
		for(int j = 0; j < Y->columns; j++)
			Y->entries[i][j] = arr[i][j];
	}
	return Y;
}