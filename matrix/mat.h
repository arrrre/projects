#include <stdbool.h>

typedef struct {
    int rows, cols;
    float* data;
} matrix_t;

matrix_t* mat_create(int rows, int cols);
matrix_t* mat_load(int rows, int cols, const char* filename);
void mat_free(matrix_t* mat);
void mat_print(matrix_t* mat);
bool mat_copy(matrix_t* dst, matrix_t* src);
void mat_clear(matrix_t* mat);
void mat_fill(matrix_t* mat, float x);
void mat_fill_rand(matrix_t* mat, float min, float max);
void mat_scale(matrix_t* mat, float scale);
float mat_sum(matrix_t* mat);
bool mat_add(matrix_t* out, const matrix_t* a, const matrix_t* b);
bool mat_sub(matrix_t* out, const matrix_t* a, const matrix_t* b);
bool mat_mul(
    matrix_t* out, const  matrix_t* a, const matrix_t* b,
    bool zero_out, bool transpose_a, bool transpose_b
);
bool mat_mul_ew(matrix_t* out, const matrix_t* a, const matrix_t* b);

bool mat_relu(matrix_t* out, const matrix_t* in);
bool mat_softmax(matrix_t* out, const matrix_t* in);
bool mat_cross_entropy(matrix_t* out, const matrix_t* p, const matrix_t* q);
bool mat_relu_add_grad(matrix_t* out, const matrix_t* in);
bool mat_softmax_add_grad(matrix_t* out, const matrix_t* in);
bool mat_cross_entropy_add_grad(matrix_t* out, const matrix_t* p, const matrix_t* q);
