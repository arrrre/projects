#include <stdio.h>

#include "base.h"
#include "arena.h"
#include "matrix.h"
#include "model.h"

void create_mnist_model(
    mem_arena* arena, model* model,
    u32 image_size, u32 label_count, u32 batch_size
);
void draw_mnist_digit(f32* data, u32 width, u32 height);
void predict_mnist_digit(
    model* m, f32* data,
    u32 cols, u32 width, u32 height
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
    matrix* test_images  = mat_load(perm_arena, TEST_COUNT, IMAGE_SIZE, "data/test_images.mat");
    matrix* train_labels = mat_create(perm_arena, TRAIN_COUNT, LABEL_COUNT);
    matrix* test_labels  = mat_create(perm_arena, TEST_COUNT, LABEL_COUNT);

    {
        matrix* train_labels_file = mat_load(perm_arena, TRAIN_COUNT, 1, "data/train_labels.mat");
        matrix* test_labels_file  = mat_load(perm_arena, TEST_COUNT, 1, "data/test_labels.mat");

        for (u32 i = 0; i < TRAIN_COUNT; i++) {
            u32 num = train_labels_file->data[i];
            train_labels->data[i * LABEL_COUNT + num] = 1.0f;
        }

        for (u32 i = 0; i < TEST_COUNT; i++) {
            u32 num = test_labels_file->data[i];
            test_labels->data[i * LABEL_COUNT + num] = 1.0f;
        }
    }

    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images  = test_images,
        .test_labels  = test_labels,

        .epochs = 10,
        .batch_size = 50,
        .learning_rate = 0.01f
    };

    draw_mnist_digit(test_images->data, IMAGE_WIDTH, IMAGE_HEIGHT);
    for (u32 i = 0; i < LABEL_COUNT; i++) {
        printf("%.0f ", test_labels->data[i]);
    }
    printf("\n\n");

    model* m = model_create(perm_arena, 10);
    create_mnist_model(
        perm_arena, m, IMAGE_SIZE,
        LABEL_COUNT, training_desc.batch_size
    );

    model_train(m, &training_desc);

    // Test on an image
    predict_mnist_digit(
        m, &test_images->data[12 * test_images->cols],
        test_images->cols, IMAGE_WIDTH, IMAGE_HEIGHT
    );

    const char* filename = "data/model.bin";
    model_save(m, filename);

    arena_destroy(perm_arena);

    return 0;
}

void create_mnist_model(
    mem_arena* arena, model* m,
    u32 image_size, u32 label_count, u32 batch_size
) {
    u32 hidden1 = 128;
    u32 hidden2 = 64;

    model_add_layer(arena, m, LAYER_LINEAR, image_size, hidden1, batch_size);
    model_add_layer(arena, m, LAYER_RELU,   hidden1,    hidden1, batch_size);
    
    model_add_layer(arena, m, LAYER_LINEAR, hidden1,    hidden2, batch_size);
    model_add_layer(arena, m, LAYER_RELU,   hidden2,    hidden2, batch_size);

    model_add_layer(arena, m, LAYER_LINEAR, hidden2,    label_count, batch_size);
    model_add_layer(arena, m, LAYER_SOFTMAX, label_count, label_count, batch_size);
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

void predict_mnist_digit(
    model* m, f32* data,
    u32 cols, u32 width, u32 height
) {
    matrix test_img = {
        .rows = 1,
        .cols = cols,
        .data = data
    };
    draw_mnist_digit(test_img.data, width, height);
    u32 guess = model_predict(m, &test_img);
    printf("Model guess: %d\n", guess);
}
