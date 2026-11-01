/*
 * CWE-78: OS Command Injection.
 * A tainted value (an environment variable) reaches a command-execution sink.
 * The hybrid scanner should: (1) have semgrep flag the system() call site,
 * then (2) have the LLM confirm the taint path env-var -> system().
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void run_user_report(void) {
    char command[256];
    /* attacker-controlled: read straight from the environment */
    const char *user = getenv("REPORT_NAME");

    snprintf(command, sizeof(command), "cat /var/reports/");
    strncat(command, user, sizeof(command) - strlen(command) - 1);

    /* SINK: the composed string is handed to the shell unvalidated */
    system(command);
}

/* Safe counterpart: the exact same sink, but fed a fixed constant. A rule
 * that only sees the sink cannot tell this apart from the function above --
 * which is the whole reason the semantic (LLM) stage exists. */
void run_fixed_report(void) {
    char command[256];
    snprintf(command, sizeof(command), "cat /var/reports/summary.txt");
    system(command);
}
