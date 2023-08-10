
#include <stdlib.h>   /* malloc and friends */
#include <stdint.h>   /* uint32_t etc */
#include <string.h>   /* memcpy */
#include <stdio.h>    /* print error message before killing process(!?!?) */

#include "../lz4/lib/lz4.h"

#ifdef USE_KCB
#include "../kcb/src/bitshuffle.h"
#else
int64_t bshuf_untrans_bit_elem(const void* in, void* out, const size_t size, 
        const size_t elem_size) ;
#endif

/* see https://justine.lol/endian.html */
#define READ32BE(p) \
  ( (uint32_t)(255 & (p)[0]) << 24 |\
    (uint32_t)(255 & (p)[1]) << 16 |\
    (uint32_t)(255 & (p)[2]) <<  8 |\
    (uint32_t)(255 & (p)[3])       )

#define READ64BE(p) \
  ( (uint64_t)(255 & (p)[0]) << 56 |\
    (uint64_t)(255 & (p)[1]) << 48 |\
    (uint64_t)(255 & (p)[2]) << 40 |\
    (uint64_t)(255 & (p)[3]) << 32 |\
    (uint64_t)(255 & (p)[4]) << 24 |\
    (uint64_t)(255 & (p)[5]) << 16 |\
    (uint64_t)(255 & (p)[6]) <<  8 |\
    (uint64_t)(255 & (p)[7])       )

#ifndef unlikely
#define unlikely(expr) (__builtin_expect ((expr),(0)))
#endif
#define BLK 8192

#define CAT_I(a,b) a##b
#define CAT(a,b) CAT_I(a, b)

#define DATATYPE uint16_t
#define NB ((int) sizeof(uint16_t))

#include "bshuf.c"

#undef DATATYPE
#undef NB

#define DATATYPE uint32_t
#define NB ((int) sizeof(uint32_t))

#include "bshuf.c"

#undef DATATYPE
#undef NB

#define DATATYPE uint8_t
#define NB ((int) sizeof(uint8_t))

#include "bshuf.c"
