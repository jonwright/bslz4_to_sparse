


int CAT( bslz4_csc_, DATATYPE) ( const char *restrict compressed,   /* compressed chunk */
                                int compressed_length, 
                                const uint8_t *restrict mask,
                                int NIJ,
                                DATATYPE *restrict outpx, 
                                uint32_t *restrict output_adr,
                                int threshold,
                                double *restrict output,
                                int NOUT,
                                float *restrict data,
                                uint32_t *restrict indices,
                                uint32_t *restrict indptr );

int CAT( bslz4_csc_, DATATYPE) (  const char *restrict compressed,   /* compressed chunk */
                                int compressed_length, 
                                const uint8_t *restrict mask,
                                int NIJ,
                                DATATYPE *restrict outpx, 
                                uint32_t *restrict output_adr,
                                int threshold,
                                double *restrict output,
                                int NOUT,
                                float *restrict data,
                                uint32_t *restrict indices,
                                uint32_t *restrict indptr ){
    size_t total_output_length;
    int blocksize, remaining, p;
    uint32_t nbytes;
    DATATYPE val, cut, tmp1[BLK/NB], tmp2[BLK/NB]; /* stack local place to decompress to */
#ifdef USE_KCB
    char scratch[BLK]; /* stack local place to shuffle (bits vs bytes) */
#endif
    int npx;     /* number of pixels written to the output */
    int i0;
    int j;
    int ret;
    unsigned int k;
    npx = 0;
    i0 = 0;
    cut = threshold;
/*    printf("Enter csc NOUT %d\n", NOUT); */
    total_output_length = READ64BE( compressed );
    if (total_output_length/NB > (uint64_t) NIJ){ 
        printf("Not enough output space, %ld %d\n", total_output_length, NIJ);
        return -99; 
    };
    blocksize = (int) READ32BE( (compressed+8) );
    if (blocksize == 0) { blocksize = BLK; }
    if(  blocksize != BLK ){
        printf("Sorry, only for 8192 internal blocks\n");
       return -101;
    }
    /* init output as 0 */
    for( j = 0; j < NOUT; j++) output[j] = 0.0;

    remaining = total_output_length;
    p = 12;
    i0 = 0;
/*    printf("Enter loop\n"); */

    for( remaining = total_output_length; remaining >= BLK; remaining = remaining - BLK){
/*        printf("remaining %d\n",remaining); */
        nbytes = READ32BE( &compressed[p] );
        ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                   (char*) &tmp1[0],
                                    nbytes,
                                    BLK );
        p = p + nbytes + 4;
        if unlikely( ret != BLK )  {
            printf("ret %d blocksize %d\n",ret, blocksize);
            printf("Returning as ret wrong size\n");
            return -2;
        }
#ifdef USE_KCB
        bitshuf_decode_block((char*) &tmp2[0],
                             (char*) &tmp1[0],
                             scratch,
                             (size_t) BLK/NB,
                             (size_t) NB);
#else
        bshuf_untrans_bit_elem((void*) tmp1, (void*) tmp2, (size_t) BLK/NB,(size_t) NB);
#endif
         /* save output      
        printf("block i0 %d\n",i0); */
        for( j = 0; j < BLK/NB; j++){
            val = tmp2[j]*mask[j + i0];
            if (unlikely(val>0)){
                /* pixel splits here */
                for( k = indptr[j+i0]; k<indptr[j+i0+1]; k++){
                    output[ indices[k] ] += (((double) data[k]) * tmp2[j]);
                }
                if unlikely( tmp2[j] > cut ) {
                    *(outpx++) = tmp2[j];
                    *(output_adr++) = j + i0;
                    npx++;
                }
            }
        }
        i0 += (BLK / NB);
    }    
    blocksize = ( 8 * NB ) * ( remaining / (8 * NB) );
    if( blocksize > 0 ){
        nbytes = READ32BE( &compressed[p] );
        ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                   (char*) tmp1,
                                    nbytes,
                                    blocksize );
        p = p + nbytes + 4;
        if unlikely( ret != blocksize )  {
            printf("ret %d blocksize %d\n",ret, blocksize);
            printf("Returning as ret wrong size\n");
            return -2;
        }
#ifdef USE_KCB
        bitshuf_decode_block((char*) &tmp2[0],
                             (char*) &tmp1[0],
                             scratch,
                             (size_t) blocksize/NB,
                             (size_t) NB);
#else
        bshuf_untrans_bit_elem((void*) tmp1, (void*) tmp2, (size_t) blocksize/NB, (size_t) NB);
#endif
    }
    remaining -= blocksize;
    if ( remaining>0 ) {
        memcpy( &tmp2[blocksize/NB], &compressed[compressed_length - remaining], remaining);
    }
         /* save output
            printf("block end i0 %d\n",i0); */
    for( j = 0; j < (remaining + blocksize)/NB; j++){
            val = tmp2[j]*mask[j + i0];
            if (unlikely(val>0)){
                /* pixel splits here */
                for( k = indptr[j+i0]; k<indptr[j+i0+1]; k++){
                    output[ indices[k] ] += (((double)data[k]) * tmp2[j]);
                }
                if unlikely( tmp2[j] > cut ) {
                    *(outpx++) = tmp2[j];
                    *(output_adr++) = j + i0;
                    npx++;
                }
             }
        }
    return npx;
}
