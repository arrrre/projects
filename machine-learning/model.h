#ifndef MODEL_H
#define MODEL_H

#include "base.h"
#include "arena.h"
#include "matrix.h"

typedef enum {
    LAYER_LINEAR,
    LAYER_RELU,
    LAYER_SOFTMAX,
} layer_type;

typedef struct {
    layer_type type;
    
    matrix* W;
    matrix* dW;
    matrix* b;
    matrix* db;

    matrix* X;
    matrix* dX;
    matrix* Y;
} model_layer;

typedef struct {
    u32 num_layers;
    u32 max_layers;
    model_layer** layers;
} model;

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

model* model_create(mem_arena* arena, u32 max_layers);
model* model_load(mem_arena* arena, const char* filename, u32 batch_size);
void model_save(model* m, const char* filename);
b32 model_add_layer(
    mem_arena* arena, model* m, layer_type type,
    u32 in_size, u32 out_size, u32 batch_size
);
void model_train(model* m, const model_training_desc* training_desc);
f32 model_evaluate(
    model* m, matrix* test_images,
    matrix* test_labels, u32 batch_size
);
u32 model_predict(model* m, matrix* image);

#endif
