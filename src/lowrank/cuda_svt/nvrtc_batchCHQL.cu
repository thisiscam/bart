#include <stdlib.h>
#include <stdio.h>
#include <nvrtc.h>
#include <cuda.h>
#include "incbin.h"
#include "nvrtc_batchCHQL.cuh"
#include "cudaBatchSVT_macros.h" 
#include "num/cuda_commons.h"
#include <iostream>
#include <assert.h>
#include "lru_cache.hpp"

INCBIN(batchCHQLSRC, "lowrank/cuda_svt/deviceCHQL.cu");
INCBIN(batchCHQLINCLUDE1, "lowrank/cuda_svt/deviceSVD.cuh");
INCBIN(batchCHQLINCLUDE2, "lowrank/cuda_svt/cudaBatchSVT_macros.h");
INCBIN(batchCHQLINCLUDE3, "lowrank/cuda_svt/cudacomplex.h");

static LRUCache<string, CUfunction*> program_cache = LRUCache<string, CUfunction*>(10);

void nvrtc_batchCHQL(singlecomplex *a, int m, float *s, 
  singlecomplex *u, float *work, int batch_size)
{
    assert(batch_size != 0 && m != 0 && "Invalid parameter");

    int NUM_BLOCKS = batch_size / INTERLEAVE + (((batch_size % INTERLEAVE) == 0) ? 0 : 1);
    int NUM_THREADS = INTERLEAVE;

 //   ((char*)gbatchCHQLSRCData)[gbatchCHQLSRCSize - 1] = '\0';

    char* jit_program_name;
    asprintf(&jit_program_name, "%s_gdim=%dbdim=%d", "deviceCHQL.cu", NUM_BLOCKS, NUM_THREADS);
    char* jit_function_name;
    asprintf(&jit_function_name, "%s_gdim%d_bdim%d", "deviceCHQL", NUM_BLOCKS, NUM_THREADS);

    CUfunction *kernel = program_cache.get(jit_function_name);
    if(kernel == NULL) {
        nvrtcProgram prog;
        const char *headers[] = {"", (char*)gbatchCHQLINCLUDE1Data, (char*)gbatchCHQLINCLUDE2Data, (char*)gbatchCHQLINCLUDE3Data};
        const char *header_names[] = {"math.h", "deviceSVD.cuh", "cudaBatchSVT_macros.h", "cudacomplex.h"};
        NVRTC_SAFE_CALL(
        nvrtcCreateProgram(&prog,         // prog
                           (char*)gbatchCHQLSRCData,         // buffer
                           jit_program_name,    // name
                           4,             // numHeaders
                           headers,          // headers
                           header_names));        // includeNames
        char* maxA_opt, *BATCH_SIZE_opt, *f_name_opt;
        asprintf(&maxA_opt, "-DmaxA=%d", m);
        asprintf(&BATCH_SIZE_opt, "-DBATCH_SIZE=%d", batch_size);
        asprintf(&f_name_opt, "-DdeviceCHQLFName=%s", jit_function_name);
        const char *opts[] = {
                              "--maxrregcount=32",
                              "-lineinfo",
                              "--include-path=/usr/local/cuda/include",
                              maxA_opt,
                              BATCH_SIZE_opt,
                              f_name_opt,
                              "-D__x86_64__"
                            };

        nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                      7,     // numOptions
                                                      opts); // options
        free(maxA_opt); free(BATCH_SIZE_opt); free(f_name_opt);

        // Obtain compilation log from the program.
        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        char *log = (char*)malloc(sizeof(char) * logSize);
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
        printf("%s\n", log);
        free(log);
        if (compileResult != NVRTC_SUCCESS) {
            exit(1);
        }

        // Obtain PTX from the program.
        size_t ptxSize;
        NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        char ptx[ptxSize];
        NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
        // Destroy the program.
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

        CUmodule module;
        kernel = new CUfunction();
        CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
        CUDA_SAFE_CALL(cuModuleGetFunction(kernel, module, jit_function_name));
        program_cache.put(jit_function_name, kernel);
    }
    free(jit_program_name); free(jit_function_name);

    // Execute kernel
    void *args[] = { &a, &m, &s, &u, &work};
    CUDA_SAFE_CALL(
        cuLaunchKernel(*kernel,
                       NUM_BLOCKS, 1, 1,   // grid dim
                       NUM_THREADS, 1, 1,    // block dim
                       0, NULL,             // shared mem and stream
                       args, 0));           // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());
}
