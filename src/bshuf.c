#define BLK 8192



int CAT( bslz4_, DATATYPE) ( const char * compressed,   /* compressed chunk */
                int compressed_length, 
                const uint8_t * mask,
                int NIJ,
                DATATYPE * output, 
                uint32_t * output_adr,
                int threshold );

int CAT( bslz4_, DATATYPE) ( const char * compressed,   /* compressed chunk */
                int compressed_length,
                const uint8_t * mask,
                int NIJ,
                DATATYPE * output, 
                uint32_t * output_adr,
                int threshold ){    
    size_t total_output_length;
    int blocksize, remaining, p;
    uint32_t nbytes;
    DATATYPE tmp1[BLK/NB], tmp2[BLK/NB]; /* stack local place to decompress to */
    int npx = 0;     /* number of pixels written to the output */
    int i0 = 0;
    int j;
    int ret;
    DATATYPE cut, val;
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
        for( j = 0; j < BLK/NB; j++){
             if unlikely(  mask[j + i0] && (tmp2[j] > cut) ){
                 output[ npx ] = tmp2[j];
                 output_adr[ npx ] = j + i0;
                 npx = npx + 1;
                 if unlikely(npx > NIJ) return -npx;
             }
        }
        i0 += (BLK / NB);
        remaining -= BLK;
    }    
    blocksize = ( 8 * NB ) * ( remaining / (8 * NB) );
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
        bshuf_untrans_bit_elem((void*) &tmp1[0], (void*) &tmp2[0], (size_t) blocksize/NB, (size_t) NB);
         /* save output */     
        for( j = 0; j < blocksize/NB; j++){
             if unlikely(  mask[j + i0] && (tmp2[j] > cut) ){
                 output[ npx ] = tmp2[j];
                 output_adr[ npx ] = j + i0;
                 npx = npx + 1;
                 if unlikely(npx > NIJ) return -npx;
             }
        }
        i0 += (blocksize / NB);
        remaining -= blocksize;
    }
    
    while( remaining > 0 ){
        val = (DATATYPE) compressed[ p ];
        if unlikely( mask[i0] && val > cut ){
          output[ npx ] = val;
          output_adr[ npx ] = i0;
          npx = npx + 1;
          if unlikely(npx > NIJ) return -npx;
        }
        p += NB;
        remaining -= NB;
        i0 += 1;
    }        
    return npx;
}
