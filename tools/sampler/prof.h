/* in-process PC sampler: SIGPROF on CPU time; PCs written to a file.
 * Build with -D_GNU_SOURCE (REG_RIP). */
#include <signal.h>
#include <sys/time.h>
#include <ucontext.h>
static uintptr_t prof_pc[4000000]; static volatile long prof_n;
static void prof_handler(int sig, siginfo_t *si, void *ctx) {
    (void) sig; (void) si; ucontext_t *uc = (ucontext_t *) ctx;
    if (prof_n < 4000000) prof_pc[prof_n++] = (uintptr_t) uc->uc_mcontext.gregs[REG_RIP];
}
static void prof_start(void) {
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = prof_handler; sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGPROF, &sa, 0);
    struct itimerval it = {{0, 100}, {0, 100}}; setitimer(ITIMER_PROF, &it, 0);
}
static void prof_stop(const char *fn) {
    struct itimerval it = {{0, 0}, {0, 0}}; setitimer(ITIMER_PROF, &it, 0);
    FILE *f = fopen(fn, "w"); for (long i = 0; i < prof_n; i++) fprintf(f, "%lx\n", (unsigned long) prof_pc[i]); fclose(f);
}
