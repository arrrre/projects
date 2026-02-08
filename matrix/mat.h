#include <stdbool.h>

typedef struct {
    int rows, cols;
    float* data;
} matrix_t;

matrix_t* mat_create(int rows, int cols);
void mat_free(matrix_t* m);
void mat_print(matrix_t* m);
void mat_clear(matrix_t* m);
void mat_fill(matrix_t* m, float f);
void mat_fill_rand(matrix_t* m, float min, float max);
void mat_scale(matrix_t* m, float f);
void mat_transpose(matrix_t* m);
bool mat_add(matrix_t* out, matrix_t* a, matrix_t* b);
bool mat_sub(matrix_t* out, matrix_t* a, matrix_t* b);
bool mat_mul(matrix_t* out, matrix_t* a, matrix_t* b, bool transpose_a, bool transpose_b);
bool mat_mul_ew(matrix_t* out, matrix_t* a, matrix_t* b);
