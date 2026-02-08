#include <stdio.h>

#include "mat.h"

int main() {
	int rows = 2, cols = 2;
	int size = rows * cols;
	matrix_t* m1 = mat_create(rows, cols);
	matrix_t* m2 = mat_create(rows, cols);
	matrix_t* m3 = mat_create(rows, cols);
	matrix_t* m4 = mat_create(rows, cols);
	matrix_t* m5 = mat_create(cols, rows);

	for (int i = 0; i < size; i++) {
		m1->data[i] = (i + 1) * 10;
		m2->data[i] = (i + 5) * 10;
	}
	// mat_fill_rand(m2, 5, 10);

	bool success_add = mat_add(m3, m1, m2);
	bool success_mul1 = mat_mul(m4, m1, m2, false, false);
	bool success_mul2 = mat_mul(m5, m1, m2, true, true);

	printf("m1\n");
	mat_print(m1);
	printf("m2\n");
	mat_print(m2);
	printf("m3 %s\n", success_add ? "success" : "failure");
	mat_print(m3);
	printf("m4 %s\n", success_mul1 ? "success" : "failure");
	mat_print(m4);
    printf("m5 %s\n", success_mul2 ? "success" : "failure");
	mat_print(m5);

	mat_free(m1);
	mat_free(m2);
	mat_free(m3);
	mat_free(m4);
	mat_free(m5);

	return 0;
}
