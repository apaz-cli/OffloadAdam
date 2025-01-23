#!/bin/sh


#if [ ! grep avx512 /proc/cpuinfo ]; then
#    echo "AVX512 not supported"
#    exit 1
#fi

gcc offload_adam.c -lm -O3 -march=native
