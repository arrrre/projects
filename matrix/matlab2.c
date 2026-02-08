#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "mat.h"

bool starts_with(const char *str, const char *pre) {
    size_t lenpre = strlen(pre), lenstr = strlen(str);
    return lenstr < lenpre ? false : memcmp(pre, str, lenpre) == 0;
}

int main() {
	char input[256];
	while (1) {
		printf(">> ");
		if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (strlen(input) == 0) continue;
        if (strcmp(input, "q") == 0) break;

		if (starts_with(input, "clear")) {
			printf("input: %s\n", input);
		} else if (starts_with(input, "close")) {
			printf("input: %s\n", input);
		}
	}

	return 0;
}
