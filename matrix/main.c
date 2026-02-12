#include <stdio.h>

#include "mat.h"

void draw_mnist_digit(float* data, int image_rows, int image_cols) {
	for (int y = 0; y < image_rows; y++) {
		for (int x = 0; x < image_cols; x++) {
			float num = data[x + y * image_cols];
			// Some grayscale color thing... 0-255
			unsigned int col = 232 + (unsigned int)(num * 24);
			printf("\x1b[48;5;%dm  ", col);
		}
		printf("\n");
	}
	printf("\x1b[0m");
}

int main() {
	const int image_rows = 28, image_cols = 28;
	const int image_size = image_rows * image_cols;
	const int train_set_size = 60000;
	const int test_set_size = 10000;
	matrix_t* train_images = mat_load(train_set_size, image_size, "data/train_images.mat");
	matrix_t* test_images = mat_load(test_set_size, image_size, "data/test_images.mat");
	matrix_t* train_labels = mat_create(train_set_size, 10);
	matrix_t* test_labels = mat_create(test_set_size, 10);
	
	{
		matrix_t* train_labels_file = mat_load(train_set_size, 1, "data/train_labels.mat");
		matrix_t* test_labels_file = mat_load(test_set_size, 1, "data/test_labels.mat");

		for (int i = 0; i < train_set_size; i++) {
			unsigned int num = train_labels_file->data[i];
			train_labels->data[i * 10 + num] = 1.0f;
		}

		for (int i = 0; i < test_set_size; i++) {
			unsigned int num = test_labels_file->data[i];
			test_labels->data[i * 10 + num] = 1.0f;
		}

		mat_free(train_labels_file);
		mat_free(test_labels_file);
	}

	draw_mnist_digit(train_images->data, image_rows, image_cols);
	for (int i = 0; i < 10; i++) {
		printf("%.0f ", train_labels->data[i]);
	}
	printf("\n");

	mat_free(train_images);
	mat_free(test_images);
	mat_free(train_labels);
	mat_free(test_labels);

	return 0;
}
