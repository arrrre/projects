#include <stdio.h>

#include "mat.h"

int main() {
    int rows = 2, cols = 3;
    int size = rows * cols;
    matrix_t* m1 = mat_create(rows, cols);
    matrix_t* m2 = mat_create(rows, cols);
    matrix_t* m3 = mat_create(rows, rows);
    for (int i = 0; i < size; i++) {
        m1->data[i] = i + 1;
        m2->data[i] = i + 2;
    }
    // 1 2 3 * 2 5 = 20 38
    // 4 5 6   3 6   47 92
    //         4 7
    mat_mul(m3, m1, m2, false, true);
    printf("m1\n");
    mat_print(m1);
    printf("m2\n");
    mat_print(m2);
    printf("m3\n");
    mat_print(m3);
}