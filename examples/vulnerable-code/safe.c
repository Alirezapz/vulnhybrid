/*
 * A deliberately SAFE file: no unbounded copies, no shell execution, no
 * non-literal format strings. A good scanner should flag nothing here --
 * it exercises the false-positive behaviour, not just the true positives.
 */
#include <stdio.h>
#include <string.h>

int sum_bounded(const int *values, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += values[i];
    }
    return total;
}

void copy_bounded(const char *src, char *dst, size_t dst_size) {
    /* bounded, size-aware copy -- not a dangerous-copy sink */
    snprintf(dst, dst_size, "%s", src);
}

void print_label(const char *label) {
    /* literal format string, value passed as an argument -- safe */
    printf("%s\n", label);
}
