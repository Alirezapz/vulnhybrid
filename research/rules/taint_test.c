#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
void external_environment(void) {
    char *command = getenv("TASK_COMMAND");
    // ruleid: explicit-source-command-taint
    system(command);
}
void constant_command(void) {
    // ok: explicit-source-command-taint
    system("date");
}
void external_read(void) {
    char command[128];
    read(0, command, sizeof(command)-1);
    command[127] = '\0';
    // todoruleid: explicit-source-command-taint
    system(command);
}
void raw_external_read(void) {
    char command[128];
    read(0, command, sizeof(command));
    // ruleid: explicit-source-command-taint
    system(command);
}
void overwritten_environment(void) {
    char *command = getenv("TASK_COMMAND");
    command = "date";
    // ok: explicit-source-command-taint
    system(command);
}
