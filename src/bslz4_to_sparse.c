
#include <stdlib.h>   /* malloc and friends */
#include <stdint.h>   /* uint32_t etc */
#include <string.h>   /* memcpy */
#include <stdio.h>    /* print error message before killing process(!?!?) */
#include <lz4.h>      /* assumes you have this already */

#include "bshuf.h"

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


int bslz4_u16 ( const char * compressed,   /* compressed chunk */
                int compressed_length, 
                const uint8_t * mask,
                int NIJ,
                uint16_t * output, 
                uint32_t * output_adr,
                int threshold );


int bslz4_u16 ( const char * compressed,   /* compressed chunk */
                int compressed_length,
                const uint8_t * mask,
                int NIJ,
                uint16_t * output, 
                uint32_t * output_adr,
                int threshold ){
    size_t total_output_length;
    int blocksize;
    int blocks_length, lastblock;
    int p = 12;      /* pointer to the next block in the input */
    uint16_t tmp1[4096], tmp2[4096]; /* stack local place to decompress to */
    int npx = 0;     /* number of pixels written to the output */
    int i0 = 0;
    int i, j;
    uint32_t nbytes;
    int ret, copied;
    uint16_t *tocopy;
    uint16_t cut;
    cut = (uint16_t) threshold;
    
    
    
    total_output_length = READ64BE( compressed );
    if (((int)total_output_length/2) > NIJ) return -301;
    blocksize = (int) READ32BE( (compressed+8) );
    if (blocksize == 0) { blocksize = 8192; }
    if(  blocksize > 8192 ){
       return -101;
    }
    
    blocks_length = (int)( (total_output_length + (size_t) blocksize - 1) / (size_t) blocksize );
    for( i = 0; i < blocks_length - 1 ; i++ ){
        nbytes = (int) READ32BE( &compressed[p] );
        ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                   (char*) &tmp1[0],
                                    nbytes,
                                    blocksize );
        if ( ret != blocksize )  {
            printf("ret %d blocksize %d\n",ret, blocksize);
            printf("Returning as ret wrong size\n");
            return -2;
        }
         /* bitshuffle here */
         bshuf_untrans_bit_elem((void*) &tmp1[0], (void*) &tmp2[0], blocksize/2, 2);
         /* point to the next chunk */
         p = p + nbytes + 4;
         /* save output */      
         for( j = 0; j < blocksize/2; j++){
             if( mask[j + i0] && (tmp2[j] > cut) ){
                 output[ npx ] = tmp2[j];
                 output_adr[ npx ] = j + i0;
                 npx = npx + 1;
             }
          }
          i0 += (blocksize / 2);
    }
    /* last block, might not be full blocksize */
    lastblock = (int) total_output_length - blocksize * (blocks_length - 1);
     /* last few bytes are copied flat */
    copied = lastblock % ( 8 * 2 );
    lastblock -= copied;
    nbytes = (int) READ32BE( &compressed[p] );
    ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                               (char*) &tmp1[0],
                               nbytes,
                               lastblock );
      if ( ret != lastblock ) return -2;
      /* bitshuffle here */
      bshuf_untrans_bit_elem((void*) &tmp1[0], (void*) &tmp2[0], lastblock/2, 2);
      for( j = 0; j < lastblock/2; j++){
        if( mask[j + i0] && (tmp2[j] > cut) ){
          output[ npx ] = tmp2[j];
          output_adr[ npx ] = j + i0;
          npx = npx + 1;
        }
      }
      i0 += lastblock / 2;
      tocopy = (uint16_t *) &compressed[ compressed_length - (size_t) copied ];
      for( j = 0; j < copied/2; j++){
        if( mask[j + i0] && (tocopy[j] > cut) ){
          output[ npx ] = tocopy[j];
          output_adr[ npx ] = j + i0;
          npx = npx + 1;
        }
      }       
    
    return npx;
}

int bslz4_u32 ( const char * compressed,   /* compressed chunk */
                        int compressed_length, 
                        const uint8_t * mask,
                        int NIJ,
                        uint32_t * output, 
                        uint32_t * output_adr,
                        int threshold );


int bslz4_u32 ( const char * compressed,   /* compressed chunk */
                        int compressed_length,
                        const uint8_t * mask,
                        int NIJ,
                        uint32_t * output, 
                        uint32_t * output_adr,
                        int threshold ){
    size_t total_output_length;
    int blocksize;
    int blocks_length, lastblock;
    int p = 12;      /* pointer to the next block in the input */
    uint32_t tmp1[2048], tmp2[2048]; /* stack local place to decompress to */
    int npx = 0;     /* number of pixels written to the output */
    int i0 = 0;
    int i, j;
    uint32_t nbytes, cut;
    int ret, copied;
    uint16_t *tocopy;
    
    cut = (uint32_t) threshold;
    
    total_output_length = READ64BE( compressed );
    if (((int)total_output_length/4) > NIJ) return -301;
    blocksize = (int) READ32BE( (compressed+8) );
    if (blocksize == 0) { blocksize = 8192; }
    if(  blocksize > 8192 ){
       return -101;
    }
    
    blocks_length = (int)( (total_output_length + (size_t) blocksize - 1) / (size_t) blocksize );
    for( i = 0; i < blocks_length - 1 ; i++ ){
        nbytes = (int) READ32BE( &compressed[p] );
        ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                   (char*) &tmp1[0],
                                    nbytes,
                                    blocksize );
        if ( ret != blocksize )  {
            printf("ret %d blocksize %d\n",ret, blocksize);
            printf("Returning as ret wrong size\n");
            return -2;
        }
         /* bitshuffle here */
         bshuf_untrans_bit_elem((void*) &tmp1[0], (void*) &tmp2[0], blocksize/4, 4);
         /* point to the next chunk */
         p = p + nbytes + 4;
         /* save output */      
         for( j = 0; j < blocksize/4; j++){
             if( mask[j + i0] && (tmp2[j] > cut) ){
                 output[ npx ] = tmp2[j];
                 output_adr[ npx ] = j + i0;
                 npx = npx + 1;
             }
          }
          i0 += (blocksize / 4);
    }
    /* last block, might not be full blocksize */
    lastblock = (int) total_output_length - blocksize * (blocks_length - 1);
     /* last few bytes are copied flat */
    copied = lastblock % ( 8 * 4 );
    lastblock -= copied;
    nbytes = (int) READ32BE( &compressed[p] );
    ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                               (char*) &tmp1[0],
                               nbytes,
                               lastblock );
      if ( ret != lastblock ) return -2;
      /* bitshuffle here */
      bshuf_untrans_bit_elem((void*) &tmp1[0], (void*) &tmp2[0], lastblock/4, 4);
      for( j = 0; j < lastblock/4; j++){
        if( mask[j + i0] && (tmp2[j] > cut) ){
          output[ npx ] = tmp2[j];
          output_adr[ npx ] = j + i0;
          npx = npx + 1;
        }
      }
      i0 += lastblock / 4;
      tocopy = (uint16_t *) & compressed[ compressed_length - (size_t) copied ];
      for( j = 0; j < copied/4; j++){
        if( mask[j + i0] && (tocopy[j] > cut) ){
          output[ npx ] = tocopy[j];
          output_adr[ npx ] = j + i0;
          npx = npx + 1;
        }
      }       
    
    return npx;
}
