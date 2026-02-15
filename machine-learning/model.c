#include <math.h>
#include <stdio.h>
#include <string.h>

#include "model.h"
#include "prng.h"

model* model_create(mem_arena* arena, u32 max_layers) {
    model* m = PUSH_STRUCT(arena, model);
    m->layers = PUSH_ARRAY(arena, model_layer*, max_layers);

    return m;
}

void model_add_layer(
    mem_arena* arena, model* m, layer_type type,
    u32 in_size, u32 out_size, u32 batch_size
) {
    model_layer* layer = PUSH_STRUCT(arena, model_layer);
    layer->type = type;

    layer->last_input  = mat_create(arena, batch_size, in_size);
    layer->last_output = mat_create(arena, batch_size, out_size);
    layer->d_input     = mat_create(arena, batch_size, in_size);

    if (type == LAYER_LINEAR) {
        layer->weights   = mat_create(arena, in_size, out_size);
        layer->d_weights = mat_create(arena, in_size, out_size);

        layer->bias      = mat_create(arena, 1, out_size);
        layer->d_bias    = mat_create(arena, 1, out_size);
    } else {
        layer->weights = layer->d_weights = NULL;
        layer->bias    = layer->d_bias    = NULL;
    }

    m->layers[m->num_layers++] = layer;
}

void model_init_weights(model* m) {
    for (u32 i = 0; i < m->num_layers; i++) {
        model_layer* layer = m->layers[i];

        if (layer->type == LAYER_LINEAR) {
            u32 n_in = layer->weights->rows;
            u32 n_out = layer->weights->cols;

            f32 bound = sqrtf(6.0f / (f32)(n_in + n_out));

            mat_fill_rand(layer->weights, -bound, bound);

            mat_clear(layer->bias);
        }
    }
}

matrix* layer_forward(model_layer* layer, matrix* input) {
    mat_copy(layer->last_input, input);

    switch (layer->type) {
        case LAYER_LINEAR:
            // Y = XW + b
            mat_mul(layer->last_output, input, layer->weights, 1, 0, 0);
            mat_add(layer->last_output, layer->last_output, layer->bias);
            break;

        case LAYER_RELU:
            // Y = max(0, X)
            mat_relu(layer->last_output, input);
            break;

        case LAYER_SOFTMAX:
            mat_softmax(layer->last_output, input);
            break;
    }

    return layer->last_output;
}

matrix* layer_backward(model_layer* layer, matrix* grad_out) {
    matrix* grad_in = layer->d_input;

    switch (layer->type) {
        case LAYER_LINEAR:
            // dW = X^T * grad_out
            mat_mul(layer->d_weights, layer->last_input, grad_out, 1, 1, 0);
            
            // dB = sum(grad_out) over rows
            mat_sum_cols(layer->d_bias, grad_out, 1);

            // dX (grad_in) = grad_out * W^T
            mat_mul(grad_in, grad_out, layer->weights, 1, 0, 1);
            break;

        case LAYER_RELU:
            // dR = grad_out * (1 if x > 0, else 0)
            mat_clear(grad_in);
            mat_relu_add_grad(grad_in, layer->last_input, grad_out);
            break;
            
        case LAYER_SOFTMAX:
            mat_copy(grad_in, grad_out);
            break;
    }

    return grad_in;
}

void layer_update(model_layer* layer, f32 lr) {
    if (layer->type == LAYER_LINEAR) {
        // W = W - learning_rate * dW
        mat_scale(layer->d_weights, lr);
        mat_sub(layer->weights, layer->weights, layer->d_weights);

        // b = b - learning_rate * db
        mat_scale(layer->d_bias, lr);
        mat_sub(layer->bias, layer->bias, layer->d_bias);
    }
}

matrix* model_forward(model* m, matrix* input) {
    matrix* current_input = input;

    for (u32 i = 0; i < m->num_layers; i++) {
        current_input = layer_forward(m->layers[i], current_input);
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

            matrix* pred = model_forward(m, x);

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

        matrix* pred = model_forward(m, x);

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

u32 model_predict(model* m, matrix* image);
