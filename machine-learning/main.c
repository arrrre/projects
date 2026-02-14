#include <math.h>
#include <stdio.h>
#include <string.h>

#include "base.h"
#include "arena.h"
#include "matrix.h"
#include "model.h"

void draw_mnist_digit(f32* data, u32 width, u32 height);
void create_mnist_model(
    mem_arena* arena, model_context* model,
    u32 image_size, u32 label_count
);

int main(void) {
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    const u32 TRAIN_COUNT  = 60000;
    const u32 TEST_COUNT   = 10000;
    const u32 IMAGE_WIDTH  = 28;
    const u32 IMAGE_HEIGHT = 28;
    const u32 IMAGE_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT;
    const u32 LABEL_COUNT  = 10;

    matrix* train_images = mat_load(perm_arena, TRAIN_COUNT, IMAGE_SIZE, "data/train_images.mat");
    matrix* test_images = mat_load(perm_arena, TEST_COUNT, IMAGE_SIZE, "data/test_images.mat");
    matrix* train_labels = mat_create(perm_arena, TRAIN_COUNT, LABEL_COUNT);
    matrix* test_labels = mat_create(perm_arena, TEST_COUNT, LABEL_COUNT);

    {
        matrix* train_labels_file = mat_load(perm_arena, TRAIN_COUNT, 1, "data/train_labels.mat");
        matrix* test_labels_file = mat_load(perm_arena, TEST_COUNT, 1, "data/test_labels.mat");

        for (u32 i = 0; i < TRAIN_COUNT; i++) {
            u32 num = train_labels_file->data[i];
            train_labels->data[i * LABEL_COUNT + num] = 1.0f;
        }

        for (u32 i = 0; i < TEST_COUNT; i++) {
            u32 num = test_labels_file->data[i];
            test_labels->data[i * LABEL_COUNT + num] = 1.0f;
        }
    }

    draw_mnist_digit(test_images->data, IMAGE_WIDTH, IMAGE_HEIGHT);
    for (u32 i = 0; i < LABEL_COUNT; i++) {
        printf("%.0f ", test_labels->data[i]);
    }
    printf("\n\n");

    model_context* model = model_create(perm_arena);
    create_mnist_model(perm_arena, model, IMAGE_SIZE, LABEL_COUNT);
    model_compile(perm_arena, model);

    memcpy(model->input->val->data, test_images->data, sizeof(f32) * IMAGE_SIZE);
    model_feedforward(model);

    printf("pre-training output: ");
    for (u32 i = 0; i < LABEL_COUNT; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }
    printf("\n");

    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,

        .epochs = 10,
        .batch_size = 50,
        .learning_rate = 0.01f
    };
    model_train(model, &training_desc);
    
    memcpy(model->input->val->data, test_images->data, sizeof(f32) * IMAGE_SIZE);
    model_feedforward(model);
    printf("post-training output: ");
    for (u32 i = 0; i < LABEL_COUNT; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }
    printf("\n");

    arena_destroy(perm_arena);

    return 0;
}

void draw_mnist_digit(f32* data, u32 width, u32 height) {
    for (u32 y = 0; y < height; y++) {
        for (u32 x = 0; x < width; x++) {
            f32 num = data[x + y * width];
            u32 col = 232 + (u32)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m\n");
}

void create_mnist_model(
    mem_arena* arena, model_context* model,
    u32 image_size, u32 label_count
) {
    model_var* input = mv_create(arena, model, 784, 1, MV_FLAG_INPUT);

    model_var* W0 = mv_create(arena, model, 16, image_size, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W1 = mv_create(arena, model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W2 = mv_create(arena, model, label_count, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    f32 bound0 = sqrtf(6.0f / (image_size + 16));
    f32 bound1 = sqrtf(6.0f / (16 + 16));
    f32 bound2 = sqrtf(6.0f / (16 + label_count));
    mat_fill_rand(W0->val, -bound0, bound0);
    mat_fill_rand(W1->val, -bound1, bound1);
    mat_fill_rand(W2->val, -bound2, bound2);

    model_var* b0 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b1 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b2 = mv_create(arena, model, label_count, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    model_var* z0_a = mv_matmul(arena, model, W0, input, 0);
    model_var* z0_b = mv_add(arena, model, z0_a, b0, 0);
    model_var* a0 = mv_relu(arena, model, z0_b, 0);

    model_var* z1_a = mv_matmul(arena, model, W1, a0, 0);
    model_var* z1_b = mv_add(arena, model, z1_a, b1, 0);
    model_var* z1_c = mv_relu(arena, model, z1_b, 0);
    model_var* a1 = mv_add(arena, model, a0, z1_c, 0);

    model_var* z2_a = mv_matmul(arena, model, W2, a1, 0);
    model_var* z2_b = mv_add(arena, model, z2_a, b2, 0);
    model_var* output = mv_softmax(arena, model, z2_b, MV_FLAG_OUTPUT);

    model_var* y = mv_create(arena, model, label_count, 1, MV_FLAG_DESIRED_OUTPUT);

    model_var* cost = mv_cross_entropy(arena, model, y, output, MV_FLAG_COST);
}
