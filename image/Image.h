#ifndef _IMAGE
#define _IMAGE
#include "../matrix/Matrix.h"

typedef struct Image Image;

#define MAX_CHAR 2048
#define IMG_SIZE 28

struct Image {
	matrix *bytes;
	int label;
};

typedef struct {
    matrix *features;
    matrix *labels;
} Batch;

Image** load_images(char *path, int imgCount);
matrix** split_images(Image **imgs, int count);
Batch **split_into_mini_batches(matrix **pair, int total_count, int batch_size);


#endif