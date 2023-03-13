


int CAT( bslz4_, DATATYPE) ( const char *restrict compressed,   /* compressed chunk */
                int compressed_length, 
                const uint8_t *restrict mask,
                int NIJ,
                DATATYPE *restrict output, 
                uint32_t *restrict output_adr,
                int threshold );

int CAT( bslz4_, DATATYPE) ( const char *restrict compressed,   /* compressed chunk */
                int compressed_length,
                const uint8_t *restrict mask,
                int NIJ,
                DATATYPE *restrict output, 
                uint32_t *restrict output_adr,
                int threshold ){    
    size_t total_output_length;
    int blocksize, remaining, p;
    uint32_t nbytes;
    DATATYPE tmp1[BLK/NB], tmp2[BLK/NB]; /* stack local place to decompress to */
    int npx;     /* number of pixels written to the output */
    int i0;
    int j;
    int ret;
    uint32_t val, cut;
    npx = 0;
    i0 = 0;
    if( threshold < 0 ){
        printf("Threshold must be positive");
        return -100;
    }
    cut = threshold;
    total_output_length = READ64BE( compressed );
    if (total_output_length/NB > (uint64_t) NIJ){ 
        printf("Not enough output space, %ld %d\n", total_output_length, NIJ);
        return -99; 
    };
    blocksize = (int) READ32BE( (compressed+8) );
    if (blocksize == 0) { blocksize = BLK; }
    if(  blocksize != BLK ){
       return -101;
    }

    remaining = total_output_length;
    p = 12;
    i0 = 0;
    for( remaining = total_output_length; remaining >= BLK; remaining = remaining - BLK){
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
        bshuf_untrans_bit_elem((void*) tmp1, (void*) tmp2, (size_t) BLK/NB,(size_t) NB);
         /* save output */     
        for( j = 0; j < BLK/NB; j++){
             val = mask[j + i0] * tmp2[j];
             if unlikely( val > cut ) {
                 output[ npx ] = tmp2[j];
                 output_adr[ npx ] = j + i0;
                 npx = npx + 1;
             }
        }
        i0 += (BLK / NB);
    }    
    blocksize = ( 8 * NB ) * ( remaining / (8 * NB) );
    remaining -= blocksize;
    if(VERBOSE)   printf("total %ld last block size %d tocopy %d",total_output_length, blocksize, remaining);
    if( blocksize ){
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
        bshuf_untrans_bit_elem((void*) tmp1, (void*) tmp2, (size_t) blocksize/NB, (size_t) NB);
    }
    remaining -= blocksize;
    if (remaining>0) memcpy( &tmp2[blocksize/NB], &compressed[compressed_length - remaining], remaining);
         /* save output */     
    for( j = 0; j < (remaining + blocksize)/NB; j++){
         val = mask[j + i0] * tmp2[j];
         if unlikely( val > cut ) {
                 output[ npx ] = tmp2[j];
                 output_adr[ npx ] = j + i0;
                 npx = npx + 1;
             }
        }
    return npx;
}
