/*
 * CWE-134: Uncontrolled format string.
 * The format argument is a bare variable whose value comes from argv.
 */
#include <stdio.h>

void log_line(const char *msg) {
    /* SINK: 'msg' is used as the format string itself. If it contains '%n'
     * or '%s', this is a format-string vulnerability. */
    printf(msg);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc > 1) {
        log_line(argv[1]);   /* tainted: comes from the command line */
    }
    return 0;
}
