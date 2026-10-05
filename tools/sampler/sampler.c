/* LD_PRELOAD PC sampler (x86_64, ppc64le): SIGPROF from a per-process CPU
 * timer every SAMPLER_US (default 100) microseconds; at exit writes the PCs
 * and /proc/self/maps to $SAMPLER_OUT.{pcs,maps}.  No ptrace, no perf. */
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>
#define NMAX 8000000
static uintptr_t *pcs; static volatile long npc;
static void handler(int sig, siginfo_t *si, void *ctx) {
    (void) sig; (void) si; ucontext_t *uc = (ucontext_t *) ctx;
    if (npc >= NMAX) return;
#if defined(__x86_64__)
    pcs[npc++] = (uintptr_t) uc->uc_mcontext.gregs[REG_RIP];
#elif defined(__powerpc64__)
    pcs[npc++] = (uintptr_t) uc->uc_mcontext.regs->nip;
#endif
}
static timer_t tid; static pid_t owner;
__attribute__((constructor)) static void start(void) {
    if (!getenv("SAMPLER_OUT")) return;
    owner = getpid();
    pcs = malloc(sizeof(uintptr_t) * NMAX);
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = handler; sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGPROF, &sa, 0);
    struct sigevent se; memset(&se, 0, sizeof se); se.sigev_notify = SIGEV_SIGNAL; se.sigev_signo = SIGPROF;
    if (timer_create(CLOCK_PROCESS_CPUTIME_ID, &se, &tid)) return;
    long us = getenv("SAMPLER_US") ? atol(getenv("SAMPLER_US")) : 100;
    struct itimerspec it = {{0, us * 1000}, {0, us * 1000}};
    timer_settime(tid, 0, &it, 0);
}
__attribute__((destructor)) static void stop(void) {
    if (!pcs || getpid() != owner) return;
    struct itimerspec it = {{0, 0}, {0, 0}}; timer_settime(tid, 0, &it, 0);
    char fn[4096]; const char *o = getenv("SAMPLER_OUT");
    snprintf(fn, sizeof fn, "%s.pcs", o); FILE *f = fopen(fn, "w");
    for (long i = 0; i < npc; i++) fprintf(f, "%lx\n", (unsigned long) pcs[i]);
    fclose(f);
    snprintf(fn, sizeof fn, "%s.maps", o); f = fopen(fn, "w"); FILE *m = fopen("/proc/self/maps", "r");
    char line[8192]; while (fgets(line, sizeof line, m)) fputs(line, f); fclose(m); fclose(f);
}
