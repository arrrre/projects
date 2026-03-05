#ifndef MODEL_H
#define MODEL_H

#include "base.h"
#include "arena.h"
#include "matrix.h"

typedef enum {
    ACT_NONE,
    ACT_RELU,
    ACT_SOFTMAX,
} model_layer_activation_type;

typedef enum {
    LAYER_LINEAR,
    LAYER_DROPOUT,
} model_layer_type;

typedef struct {
    model_layer_type layer_type;
    model_layer_activation_type activation_type;
    
    matrix* W; matrix* dW;
    matrix* b; matrix* db;

    matrix* X; matrix* dX;
    matrix* Z; matrix* dZ;
    matrix* Y;

    f32 dropout_rate;
    matrix* dropout_mask;
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
b32 model_add_layer(
    mem_arena* arena, model* m, model_layer_type layer_type,
    model_layer_activation_type activation_type,
    u32 in_size, u32 out_size, u32 batch_size, f32 dropout_rate
);
void model_train(model* m, const model_training_desc* training_desc);
f32 model_evaluate(
    model* m, matrix* test_images,
    matrix* test_labels, u32 batch_size
);
u32 model_predict(model* m, matrix* image);

#endif
