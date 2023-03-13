#define BLK 8192



int CAT( bslz4_, DATATYPE) ( const char *restrict compressed,   /* compressed chunk */
                            int compressed_length, 
                            const uint8_t *restrict mask,
                            int NIJ,
                            DATATYPE *restrict output, 
                            uint16_t *restrict islow,
                            uint16_t *restrict jfast,
                            int threshold,
                            int nfast);

int CAT( bslz4_, DATATYPE) ( const char *restrict compressed,   /* compressed chunk */
                int compressed_length,
                const uint8_t *restrict mask,
                int NIJ,
                DATATYPE *restrict output, 
                uint16_t *restrict islow,
                uint16_t *restrict jfast,
                int threshold,
                            int nfast ) {
    size_t total_output_length, bret;
    int blocksize, remaining, p;
    uint32_t nbytes;
    DATATYPE tmp1[BLK/NB], tmp2[BLK/NB]; /* stack local place to decompress to */
    int npx = 0;     /* number of pixels written to the output */
    int i0 = 0;
    int j;
    int ret;
    DATATYPE cut;
    if (VERBOSE) printf("nfast %d %d\n", nfast, NFAST);
    cut = (DATATYPE) threshold;
        
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
    while ( remaining >= BLK ){
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
        bshuf_untrans_bit_elem((void*) &tmp1[0], (void*) &tmp2[0], (size_t) BLK/NB,(size_t) NB);
         /* save output */ 
        if likely(nfast == NFAST){
             for( j = 0; j < BLK/NB; j++){
                 if unlikely( (mask[j + i0] * tmp2[j]) > cut) {
                     output[ npx ] = tmp2[j];
                     islow[npx] = ( j + i0 )/NFAST;
                     jfast[npx] = ( j + i0 )%NFAST;
                     npx = npx + 1;
                 }
            }  
        } else {
             for( j = 0; j < BLK/NB; j++){
                 if unlikely( (mask[j + i0] * tmp2[j]) > cut) {
                     output[ npx ] = tmp2[j];
                     islow[npx] = ( j + i0 )/nfast;
                     jfast[npx] = ( j + i0 )%nfast;
                     npx = npx + 1;
                 }
            }  
        }
        i0 += (BLK / NB);
        remaining -= BLK;
    }    
    blocksize = ( 8 * NB ) * ( remaining / (8 * NB) );
    remaining -= blocksize;
    if(VERBOSE)   printf("total %ld last block size %d tocopy %d",total_output_length, blocksize, remaining);
    if( blocksize ){
        nbytes = READ32BE( &compressed[p] );
        ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                   (char*) &tmp1[0],
                                    nbytes,
                                    blocksize );
        p = p + nbytes + 4;
        if unlikely( ret != blocksize )  {
            printf("ret %d blocksize %d\n",ret, blocksize);
            printf("Returning as ret wrong size\n");
            return -2;
        }
        bret = bshuf_untrans_bit_elem((void*) &tmp1[0], (void*) &tmp2[0], (size_t) blocksize/NB, (size_t) NB);
        if(VERBOSE)   printf("ret %d bret %ld blocksize %d\n",ret, bret,blocksize);
    }
    if( remaining > 0 ) {
            memcpy( &tmp2[blocksize/NB], &compressed[ compressed_length - remaining ], remaining );
            if(VERBOSE) printf("memcopy %d\n", remaining);
    }
    if ( (blocksize + remaining) > 0 ){    
        /* save output */
        if likely(nfast == NFAST){
             for( j = 0; j < (blocksize + remaining)/NB; j++){
                 if unlikely( (mask[j + i0] * tmp2[j]) > cut) {
                     output[ npx ] = tmp2[j];
                     islow[npx] = ( j + i0 )/NFAST;
                     jfast[npx] = ( j + i0 )%NFAST;
                     npx = npx + 1;
                 }
            }  
        } else {
             for( j = 0; j < (blocksize + remaining)/NB; j++){
                 if unlikely( (mask[j + i0] * tmp2[j]) > cut) {
                     output[ npx ] = tmp2[j];
                     islow[npx] = ( j + i0 )/nfast;
                     jfast[npx] = ( j + i0 )%nfast;
                     npx = npx + 1;
                 }
            }  
        }
    }
    return npx;
}
