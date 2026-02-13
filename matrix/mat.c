#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mat.h"
#include "rand.h"

matrix_t* mat_create(int rows, int cols) {
    int size = rows * cols;
    matrix_t* m = (matrix_t*) malloc(sizeof(matrix_t));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*) malloc(size * sizeof(float));
    return m;
}

matrix_t* mat_load(int rows, int cols, const char* filename) {
    matrix_t* mat = mat_create(rows, cols);

    FILE* f = fopen(filename, "rb");

    fseek(f, 0, SEEK_END);
    int size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size = fmin(size, sizeof(float) * rows * cols);

    fread(mat->data, 1, size, f);

    return mat;
}

void mat_free(matrix_t* mat) {
    free(mat->data);
    free(mat);
}

void mat_print(matrix_t* mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%.2f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
}

bool mat_copy(matrix_t* dst, matrix_t* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) return false;

    int size = dst->rows * dst->cols;
    for (int i = 0; i < size; i++) {
        dst->data[i] = src->data[i];
    }

    return true;
}

void mat_clear(matrix_t* mat) {
    int size = mat->rows * mat->cols;
    memset(mat->data, 0, size * sizeof(float));
}

void mat_fill(matrix_t* mat, float x) {
    int size = mat->rows * mat->cols;
    for (int i = 0; i < size; i++) {
        mat->data[i] = x;
    }
}

void mat_fill_rand(matrix_t* mat, float min, float max) {
	int size = mat->rows * mat->cols;
	for (int i = 0; i < size; i++) {
		mat->data[i] = rand_get(min, max);
	}
}

void mat_scale(matrix_t* mat, float scale) {
    int size = mat->rows * mat->cols;
    for (int i = 0; i < size; i++) {
        mat->data[i] *= scale;
    }
}

float mat_sum(matrix_t* mat) {
    int size = mat->rows * mat->cols;

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += mat->data[i];
    }

    return sum;
}

bool mat_add(matrix_t* out, const matrix_t* a, const matrix_t* b) {
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (a->rows != out->rows || a->cols != out->cols) return false;

    int size = out->rows * out->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

bool mat_sub(matrix_t* out, const matrix_t* a, const matrix_t* b) {
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (a->rows != out->rows || a->cols != out->cols) return false;

    int size = out->rows * out->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

void _mat_mul_nn(matrix_t* out, const matrix_t* a, const matrix_t* b) {
    for (int i = 0; i < out->rows; i++) {
        for (int k = 0; k < a->cols; k++) {
            for (int j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] *
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_nt(matrix_t* out, const matrix_t* a, const matrix_t* b) {
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            for (int k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] *
                    b->data[k + j * b->cols];
            }
        }
    }
}

void _mat_mul_tn(matrix_t* out, const matrix_t* a, const matrix_t* b) {
    for (int k = 0; k < a->cols; k++) {
        for (int i = 0; i < out->rows; i++) {
            for (int j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] *
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_tt(matrix_t* out, const matrix_t* a, const matrix_t* b) {
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            for (int k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] *
                    b->data[k + j * b->cols];
            }
        }
    }
}

bool mat_mul(
    matrix_t* out, const  matrix_t* a, const matrix_t* b,
    bool zero_out, bool transpose_a, bool transpose_b
) {
    int a_rows = transpose_a ? a->cols : a->rows;
    int a_cols = transpose_a ? a->rows : a->cols;
    int b_rows = transpose_b ? b->cols : b->rows;
    int b_cols = transpose_b ? b->rows : b->cols;
    if (a_cols != b_rows) { return false; }
    if (a_rows != out->rows || b_cols != out->cols) { return false; }

    if (zero_out) { mat_clear(out); }

    unsigned int transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0b00: { _mat_mul_nn(out, a, b); } break;
        case 0b01: { _mat_mul_nt(out, a, b); } break;
        case 0b10: { _mat_mul_tn(out, a, b); } break;
        case 0b11: { _mat_mul_tt(out, a, b); } break;
    }

    return true;
}

bool mat_mul_ew(matrix_t* out, const matrix_t* a, const matrix_t* b) {
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (a->rows != out->rows || a->cols != out->cols) return false;

    int size = out->rows * out->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }

    return true;
}

bool mat_relu(matrix_t* out, const matrix_t* in) {
    if (out->rows != in->rows || out->cols != in->cols) return false;

    int size = out->rows * out->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = fmax(0.0f, in->data[i]);
    }

    return true;
}

bool mat_softmax(matrix_t* out, const matrix_t* in) {
    if (out->rows != in->rows || out->cols != in->cols) return false;

    int size = out->rows * out->cols;

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i]);
        sum += in->data[i];
    }

    mat_scale(out, 1.0f / sum);

    return true;
}

bool mat_cross_entropy(matrix_t* out, const matrix_t* p, const matrix_t* q) {
    if (p->rows != q->rows || p->cols != q->cols) return false;
    if (p->rows != out->rows || p->cols != out->cols) return false;

    int size = out->rows * out->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = p->data[i] == 0.0f ?
            0.0f : p->data[i] * -logf(q->data[i]);
    }

    return true;
}

bool mat_relu_add_grad(matrix_t* out, const matrix_t* in, const matrix_t* grad) {

}

bool mat_softmax_add_grad(
    matrix_t* out, const matrix_t* softmax_out, const matrix_t* grad
) {

}

bool mat_cross_entropy_add_grad(
    matrix_t* p_grad, matrix_t* q_grad, 
    const matrix_t* p, const matrix_t* q, const matrix_t* grad
) {

}
