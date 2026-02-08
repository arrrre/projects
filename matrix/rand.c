#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#include "rand.h"

bool is_initialized = false;

void rand_init() {
	srand(time(NULL));
	is_initialized = true;
}

float rand_get(float min, float max) {
	if (!is_initialized) rand_init();

	return min + ((float)rand() / RAND_MAX) * (max - min);
}
