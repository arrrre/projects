#include <math.h>
#include <stdio.h>
#include <string.h>

#include "model.h"
#include "prng.h"

model* model_create(mem_arena* arena, u32 max_layers) {
    model* m = PUSH_STRUCT(arena, model);
    m->layers = PUSH_ARRAY(arena, model_layer*, max_layers);
    m->num_layers = 0;
    m->max_layers = max_layers;

    return m;
}

b32 model_add_layer(
    mem_arena* arena, model* m, model_layer_type layer_type,
    model_layer_activation_type activation_type,
    u32 in_size, u32 out_size, u32 batch_size, f32 dropout_rate
) {
    if (m->num_layers >= m->max_layers) { return false; }

    model_layer* layer = PUSH_STRUCT(arena, model_layer);
    layer->layer_type = layer_type;
    layer->activation_type = activation_type;
    layer->dropout_rate = dropout_rate;

    layer->X  = mat_create(arena, batch_size, in_size);
    layer->dX = mat_create(arena, batch_size, in_size);
    layer->Z  = mat_create(arena, batch_size, out_size);
    layer->dZ = mat_create(arena, batch_size, out_size);
    layer->Y  = mat_create(arena, batch_size, out_size);

    if (layer_type == LAYER_LINEAR) {
        layer->W  = mat_create(arena, in_size, out_size);
        layer->dW = mat_create(arena, in_size, out_size);

        layer->b  = mat_create(arena, 1, out_size);
        layer->db = mat_create(arena, 1, out_size);

        layer->dropout_mask = NULL;
    } else if (layer_type == LAYER_DROPOUT) {
        layer->W = layer->dW = NULL;
        layer->b = layer->db = NULL;

        layer->dropout_mask = mat_create(arena, batch_size, in_size);
    }

    m->layers[m->num_layers++] = layer;
    
    return true;
}

void model_init_weights(model* m) {
    for (u32 i = 0; i < m->num_layers; i++) {
        model_layer* layer = m->layers[i];

        if (layer->layer_type == LAYER_LINEAR) {
            u32 n_in = layer->W->rows;
            f32 std;

            if (layer->activation_type == ACT_RELU) {
                std = sqrtf(2.0f / (f32)n_in);
            } else {
                u32 n_out = layer->W->cols;
                std = sqrtf(2.0f / (f32)(n_in + n_out));
            }

            f32 bound = std * sqrtf(3.0f);
            mat_fill_rand(layer->W, -bound, bound);

            mat_clear(layer->b);
        }
    }
}

matrix* layer_forward(model_layer* layer, matrix* input, b32 is_training) {
    mat_copy(layer->X, input);

    if (layer->layer_type == LAYER_LINEAR) {
        // Z = X * W + b
        mat_mul(layer->Z, layer->X, layer->W, 1, 0, 0);
        mat_add(layer->Z, layer->Z, layer->b);

        // Y = f(Z)
        switch (layer->activation_type) {
            case ACT_RELU:    mat_relu(layer->Y, layer->Z); break;
            case ACT_SOFTMAX: mat_softmax(layer->Y, layer->Z); break;
            case ACT_NONE:    mat_copy(layer->Y, layer->Z); break;
        }
    } else if (layer->layer_type == LAYER_DROPOUT) {
        if (is_training) {
            f32 scale = 1.0f / (1.0f - layer->dropout_rate);
            for (u32 i = 0; i < input->rows * input->cols; i++) {
                f32 r = prng_randf(); 
                if (r < layer->dropout_rate) {
                    layer->dropout_mask->data[i] = 0.0f;
                    layer->Y->data[i] = 0.0f;
                } else {
                    layer->dropout_mask->data[i] = scale;
                    layer->Y->data[i] = input->data[i] * scale;
                }
            }
        } else {
            mat_copy(layer->Y, input);
        }
    }

    return layer->Y;
}

matrix* layer_backward(model_layer* layer, matrix* grad_out) {
    if (layer->layer_type == LAYER_LINEAR) {
        switch (layer->activation_type) {
            case ACT_RELU:
                mat_clear(layer->dZ);
                mat_relu_add_grad(layer->dZ, layer->Z, grad_out); 
                break;
            case ACT_SOFTMAX:
            case ACT_NONE:
                mat_copy(layer->dZ, grad_out);
                break;
        }
        // dW = X^T * dZ
        mat_mul(layer->dW, layer->X, layer->dZ, 1, 1, 0);    
        // db = sum(dZ) over rows
        mat_sum_cols(layer->db, layer->dZ, 1);
        // dX (grad_in for next layer) = dZ * W^T
        mat_mul(layer->dX, layer->dZ, layer->W, 1, 0, 1);
    } else if (layer->layer_type == LAYER_DROPOUT) {
        mat_mul_ew(layer->dX, grad_out, layer->dropout_mask);
    }

    return layer->dX;
}

void layer_update(model_layer* layer, f32 lr) {
    if (layer->layer_type == LAYER_LINEAR) {
        // W = W - learning_rate * dW
        mat_scale(layer->dW, lr);
        mat_sub(layer->W, layer->W, layer->dW);

        // b = b - learning_rate * db
        mat_scale(layer->db, lr);
        mat_sub(layer->b, layer->b, layer->db);
    }
}

matrix* model_forward(model* m, matrix* input, b32 is_training) {
    matrix* current_input = input;

    for (u32 i = 0; i < m->num_layers; i++) {
        current_input = layer_forward(m->layers[i], current_input, is_training);
    }
    
    return current_input;
}

void model_backward(model* m, matrix* loss_grad) {
    matrix* current_grad = loss_grad;

    for (i64 i = m->num_layers - 1; i >= 0; i--) {
        current_grad = layer_backward(m->layers[i], current_grad);
    }
}

void model_update(model* m, f32 lr) {
    for (u32 i = 0; i < m->num_layers; i++) {
        layer_update(m->layers[i], lr);
    }
}

void get_batch(matrix* dst, matrix* src, u32 batch_idx, u32 batch_size) {
    u32 start_row = batch_idx * batch_size;
    u64 elements_per_row = src->cols;
    
    memcpy(dst->data, &src->data[start_row * elements_per_row], 
           batch_size * elements_per_row * sizeof(f32));
}

void get_batch_shuffled(
    matrix* dst, matrix* src,
    u32* training_order, 
    u32 batch_idx, u32 batch_size
) {
    u32 start_idx = batch_idx * batch_size;
    
    for (u32 i = 0; i < batch_size; i++) {
        u32 original_row = training_order[start_idx + i];
        
        memcpy(&dst->data[i * src->cols], 
               &src->data[original_row * src->cols], 
               src->cols * sizeof(f32));
    }
}

void shuffle_array(u32* array, u32 n) {
    for (u32 i = n - 1; i > 0; i--) {
        u32 j = prng_rand() % (i + 1);
        
        u32 tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

void model_train(model* m, const model_training_desc* training_desc) {
    matrix* train_images = training_desc->train_images;
    matrix* train_labels = training_desc->train_labels;
    matrix* test_images = training_desc->test_images;
    matrix* test_labels = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;

    u32 batch_size = training_desc->batch_size;
    u32 num_batches = num_examples / batch_size;

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    model_init_weights(m);

    matrix* x = mat_create(scratch.arena, batch_size, input_size);
    matrix* y = mat_create(scratch.arena, batch_size, output_size);
    matrix* loss_grad = mat_create(scratch.arena, batch_size, output_size);

    u32* training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
    for (u32 i = 0; i < num_examples; i++) {
        training_order[i] = i;
    }

    for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
        shuffle_array(training_order, num_examples);

        f32 epoch_loss = 0.0f;

        for (u32 batch = 0; batch < num_batches; batch++) {
            get_batch_shuffled(x, train_images, training_order, batch, batch_size);
            get_batch_shuffled(y, train_labels, training_order, batch, batch_size);

            matrix* pred = model_forward(m, x, true);

            epoch_loss += mat_cross_entropy(y, pred);

            mat_sub(loss_grad, pred, y);
            mat_scale(loss_grad, 1.0f / batch_size);

            model_backward(m, loss_grad);

            model_update(m, training_desc->learning_rate);
            
            printf(
                "Epoch %2d / %2d, Batch %4d / %4d, Epoch Loss: %.4f\r",
                epoch + 1, training_desc->epochs,
                batch + 1, num_batches, epoch_loss / (batch + 1)
            );
            fflush(stdout);
        }
        printf("\n");

        model_evaluate(m, test_images, test_labels, batch_size);
    }

    arena_scratch_release(scratch);
}

f32 model_evaluate(model* m, matrix* test_images, matrix* test_labels, u32 batch_size) {
    u32 num_tests = test_images->rows;
    u32 input_size = test_images->cols;
    u32 output_size = test_labels->cols;
    
    u32 num_batches = num_tests / batch_size;
    u32 num_correct = 0;

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);
    matrix* x = mat_create(scratch.arena, batch_size, input_size);
    matrix* y = mat_create(scratch.arena, batch_size, output_size);

    for (u32 batch = 0; batch < num_batches; batch++) {
        get_batch(x, test_images, batch, batch_size); 
        get_batch(y, test_labels, batch, batch_size);

        matrix* pred = model_forward(m, x, false);

        for (u32 i = 0; i < batch_size; i++) {
            num_correct += mat_argmax_row(pred, i) == mat_argmax_row(y, i);
        }
    }

    f32 accuracy = (f32)num_correct / num_tests;
    printf("Test Completed. Accuracy: %5d / %5d (%.1f%%)\n",
            num_correct, num_tests, accuracy * 100.0f
    );

    arena_scratch_release(scratch);

    return accuracy;
}

u32 model_predict(model* m, matrix* image) {
    matrix* pred = model_forward(m, image, false);

    return mat_argmax_row(pred, 0);
}
