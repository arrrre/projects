#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mat.h"

matrix_t* mat_create(int rows, int cols) {
    int size = rows * cols;
    matrix_t* m = (matrix_t*) malloc(sizeof(matrix_t));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*) malloc(size * sizeof(float));
    return m;
}

void mat_destroy(matrix_t* m) {
    free(m->data);
    free(m);
}

void mat_clear(matrix_t* m) {
    int size = m->rows * m->cols;
    memset(m->data, 0, size * sizeof(float));
}

void mat_print(matrix_t* m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.2f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
}

void mat_fill(matrix_t* m, float f) {
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        m->data[i] = f;
    }
}

void mat_scale(matrix_t* m, float f) {
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        m->data[i] = f * m->data[i];
    }
}

void mat_transpose(matrix_t* m) {
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        m->data[i] = m->data[i];
    }
}

bool mat_add(matrix_t* out, matrix_t* a, matrix_t* b) {
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (a->rows != out->rows || a->cols != out->cols) return false;

    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

bool mat_sub(matrix_t* out, matrix_t* a, matrix_t* b) {
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (a->rows != out->rows || a->cols != out->cols) return false;

    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

bool mat_mul(matrix_t* out, matrix_t* a, matrix_t* b, bool transpose_a, bool transpose_b) {
    int a_rows = transpose_a ? a->cols : a->rows;
    int a_cols = transpose_a ? a->rows : a->cols;
    int b_rows = transpose_b ? b->cols : b->rows;
    int b_cols = transpose_b ? b->rows : b->cols;
    if (a_cols != b_rows) return false;
    if (a_rows != out->rows || b_cols != out->cols) return false;

    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; k++) {
                // Determine indices based on transposition
                int idx_a = transpose_a ? (k * a->cols + i) : (i * a->cols + k);
                int idx_b = transpose_b ? (j * b->cols + k) : (k * b->cols + j);
                sum += a->data[idx_a] * b->data[idx_b];
            }
            out->data[i * out->cols + j] = sum;
        }
    }

    return true;
}

bool mat_mul_ew(matrix_t* out, matrix_t* a, matrix_t* b) {
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (a->rows != out->rows || a->cols != out->cols) return false;

    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }

    return true;
}
