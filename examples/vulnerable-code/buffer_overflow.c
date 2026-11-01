/*
 * CWE-120/121: classic unbounded copy into a fixed stack buffer.
 */
#include <string.h>
#include <stdio.h>

void greet(const char *name) {
    char buffer[16];
    /* SINK: strcpy has no destination-size bound; a long 'name' overflows
     * 'buffer'. 'name' is a parameter with no visible caller here, so the
     * model should lower its confidence rather than assume the worst. */
    strcpy(buffer, name);
    printf("Hello, %s\n", buffer);
}

void append_ext(char *path) {
    /* SINK: strcat with no bound. */
    strcat(path, ".bak");
}
