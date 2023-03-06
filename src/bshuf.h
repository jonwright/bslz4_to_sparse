
#ifndef BSHUF_H
#define BSHUF_H

#include "bitshuffle_core.h"

int64_t bshuf_trans_bit_elem(const void* in, void* out, const size_t size, 
        const size_t elem_size) ;

int64_t bshuf_untrans_bit_elem(const void* in, void* out, const size_t size, 
        const size_t elem_size) ;

#endif