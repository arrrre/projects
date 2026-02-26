#ifndef MATRIX_H
#define MATRIX_H

#include "base.h"
#include "arena.h"

typedef struct {
    u32 rows, cols;
    // Row-major
    f32* data;
} matrix;

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename);
b32 mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, f32 x);
void mat_fill_rand(matrix* mat, f32 lower, f32 upper);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
b32 mat_sum_rows(matrix* out, matrix* in, b8 zero_out);
b32 mat_sum_cols(matrix* out, matrix* in, b8 zero_out);
u64 mat_argmax(matrix* mat);
u64 mat_argmax_row(matrix* mat, u32 r);
u64 mat_argmax_col(matrix* mat, u32 c);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);
b32 mat_mul_ew(matrix* out, const matrix* a, const matrix* b);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
f32 mat_cross_entropy(const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);

#endif
