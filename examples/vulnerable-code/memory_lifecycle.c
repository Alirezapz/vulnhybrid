/*
 * Memory-lifecycle vulnerabilities (CWE-415 / CWE-416 / CWE-562).
 *
 * These are the flaw families the thesis measured a code-LLM to be WEAKEST on
 * (e.g. CWE-401 memory leak recall 0.00, CWE-415 double-free acc 0.31),
 * because they require tracking a resource's STATE across statements and
 * branches rather than following a single tainted data flow. Static
 * value-/pattern-flow analysis is comparatively good here -- which is exactly
 * why adding these rules complements, rather than duplicates, the LLM stage.
 *
 * Each bad function is paired with a safe counterpart so the rules can be
 * checked for false positives, not just recall (the same discipline the
 * pipeline's Phase-2 rules were built with).
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct node { struct node *next; int value; };

/* ---- CWE-415: Double Free ------------------------------------------------ */

void bad_double_free(int n) {
    char *buf = malloc(n);
    if (!buf) return;
    memset(buf, 0, n);
    free(buf);
    /* ... later, on the same path, freed again with no reassignment ... */
    free(buf);              /* BAD: CWE-415 */
}

void safe_realloc_then_free(int n) {
    char *buf = malloc(n);
    free(buf);
    buf = malloc(n * 2);    /* reassigned to a NEW allocation */
    free(buf);              /* safe: different object */
}

/* ---- CWE-416: Use After Free --------------------------------------------- */

void bad_use_after_free_field(struct node *p) {
    free(p);
    p->next = NULL;         /* BAD: CWE-416 (field access after free) */
}

int bad_use_after_free_deref(int *p) {
    free(p);
    return *p;              /* BAD: CWE-416 (dereference after free) */
}

int safe_use_before_free(int *p) {
    int v = *p;             /* used BEFORE free -- safe */
    free(p);
    return v;
}

/* ---- CWE-562: Return of Stack Variable Address --------------------------- */

char *bad_return_stack_buffer(void) {
    char buf[64];
    snprintf(buf, sizeof(buf), "temporary");
    return buf;             /* BAD: CWE-562 (buf dies at return) */
}

int *bad_return_local_address(void) {
    int x = 42;
    return &x;              /* BAD: CWE-562 */
}

char *safe_return_heap(void) {
    char *buf = malloc(64);
    if (buf) snprintf(buf, 64, "temporary");
    return buf;             /* safe: heap outlives the call */
}
