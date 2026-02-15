#include <stdio.h>

#include "base.h"
#include "arena.h"
#include "matrix.h"
#include "model.h"

void draw_mnist_digit(f32* data, u32 width, u32 height);
void predict_mnist_digit(
    model* m, f32* data,
    u32 cols, u32 width, u32 height
);

int main(void) {
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    const u32 TEST_COUNT   = 10000;
    const u32 IMAGE_WIDTH  = 28;
    const u32 IMAGE_HEIGHT = 28;
    const u32 IMAGE_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT;

    matrix* test_images  = mat_load(perm_arena, TEST_COUNT, IMAGE_SIZE, "data/test_images.mat");

    const char* filename = "data/model.bin";
    model* m = model_load(perm_arena, filename, 1);

    // Test on some images
    for (u32 i = 0; i < 10; i++) {
        predict_mnist_digit(
            m, &test_images->data[i * test_images->cols],
            test_images->cols, IMAGE_WIDTH, IMAGE_HEIGHT
        );
    }

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
