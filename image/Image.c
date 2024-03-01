#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Image.h"

//load images from  csv
Image** load_images_csv(char *path, int imgCount){
	FILE *fp = fopen(path, "r");
	Image **imgs = (Image**)calloc(imgCount, sizeof(Image*));
	char row[MAX_CHAR];
	int i = 0;

	while(!feof(fp) && i < imgCount){
		imgs[i] = (Image*)malloc(sizeof(Image));
		imgs[i]->bytes = createMatrix(IMG_SIZE, IMG_SIZE, false);

		fgets(row, MAX_CHAR, fp);
		char *token = strtok(row, ","); //Split row by ',' to individual tokens

		int j = 0;
		//Traverse all tokens untill row end
		while (token != NULL) {
			if (j == 0) {
				imgs[i]->label = atoi(token); //first token is the label
			}else {
				//map each pixel to its corresponding (i,j) place in the image matrice
				imgs[i]->bytes->entries[(j-1) / IMG_SIZE][(j-1) % IMG_SIZE] = atoi(token) / 256.0;
			}
			token = strtok(NULL, ","); //get next token
			j++;
		}
		i++;
	}

	fclose(fp);
	return imgs;
}

//split images to input matrices and label matrices
matrix** split_images(Image **imgs, int count){
	matrix **pair = (matrix**)malloc(sizeof(matrix*) * 2);
	pair[0] = createMatrix(IMG_SIZE * IMG_SIZE, count, false); //features vector X
	pair[1] = createMatrix(10, count, false);//features labels Y

	for(int i = 0; i < count; i++){
		pair[1]->entries[imgs[i]->label][i] = 1;

		for(int j = 0; j < pair[0]->rows; j++){
			pair[0]->entries[j][i] = imgs[i]->bytes->entries[j / IMG_SIZE][j  % IMG_SIZE];
		}
	}
	return pair;
}

//split the set into mini batches
Batch **split_into_mini_batches(matrix **pair, int total_count, int batch_size) {
	int num_batches = total_count / batch_size;
	Batch **batches = (Batch**)malloc(sizeof(Batch*) * num_batches);

	for (int i = 0; i < num_batches; i++) {
		batches[i] = (Batch*)malloc(sizeof(Batch));

		// Calculate the start and end indices for the current batch
		int start_idx = i * batch_size;
		int end_idx = (i + 1) * batch_size;

		// Create matrices for features and labels in the current batch
		batches[i]->features = createMatrix(IMG_SIZE * IMG_SIZE, batch_size, false);
		batches[i]->labels = createMatrix(10, batch_size, false);

		// Copy data from the original pair to the current batch
		for (int j = start_idx; j < end_idx; j++) {
			// Copy features
			for (int k = 0; k < pair[0]->rows; k++) {
					batches[i]->features->entries[k][j - start_idx] = pair[0]->entries[k][j];
			}

			// Copy labels
			for (int k = 0; k < pair[1]->rows; k++) {
					batches[i]->labels->entries[k][j - start_idx] = pair[1]->entries[k][j];
			}
		}
	}

	return batches;
}
