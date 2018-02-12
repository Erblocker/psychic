/*
 Copyright (c) 2017 cgoxopx(Yu).
 All rights reserved.
 
 Redistribution and use in source and binary forms are permitted
 provided that the above copyright notice and this paragraph are
 duplicated in all such forms and that any documentation,
 advertising materials, and other materials related to such
 distribution and use acknowledge that the software was developed
 by the copyright holder. The name of the
 copyright holder may not be used to endorse or promote products derived
 from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */
 
#ifndef __PS_CL_H
#define __PS_CL_H
  #include <CL/cl.h>
  
  typedef struct{
    cl_context       context;
    cl_command_queue commandQueue;
    cl_device_id     device;
  }PSCLContext;
  
  typedef struct{
    cl_program    program;
    PSCLContext * context;
  }PSCLProgram;
  
  typedef enum{
    PSCLMemRead,
    PSCLMemWrite,
    PSCLMemRDWR
  }PSCLMemMethod;
  
  struct PSCLKernel_s;
  
  typedef struct PSCLMem_s{
    void * data;
    cl_mem rem;
    size_t size;
    struct PSCLKernel_s * kernel;
    PSCLMemMethod method;
    struct PSCLMem_s * next;
  }PSCLMem;
  
  typedef struct PSCLKernel_s{
    cl_kernel     kernel;
    PSCLProgram * program;
    PSCLMem     * args;
    size_t        argn;
    cl_uint       dim;
    size_t      * globalThreads,
                * localThreads;
  }PSCLKernel;
  
  PSCLContext * PSCLCreate();
  PSCLContext * PSCLCreateContext();
  int PSCLCreateCommandQueue(PSCLContext*);
  PSCLProgram * PSCLCreateProgram(PSCLContext*,const char*);
  PSCLKernel  * PSCLCreateKernel(PSCLProgram*,const char *);
  
  PSCLMem * PSCLMemAdd(PSCLKernel*,void*,size_t,PSCLMemMethod);
  void PSCLUpdateRead(PSCLMem*);
  void PSCLUpdateUpload(PSCLMem*);
  
  void PSCLKernelSetDim(PSCLKernel*,size_t);
  void PSCLKernelSetGbThread(PSCLKernel*,const size_t*);
  void PSCLKernelSetLCThread(PSCLKernel*,const size_t*);
  void PSCLKernelSetLCTAuto(PSCLKernel*);
  
  cl_uint PSCLKernelExec(PSCLKernel*);
  
  void PSCLDestroyContext(PSCLContext*);
  void PSCLDestroyProgram(PSCLProgram*);
  void PSCLDestroyKernel(PSCLKernel*);
  
  void PSCLInit();
  void PSCLDestroy();
  int PSCLUseable();
#endif
