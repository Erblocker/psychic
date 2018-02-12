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

#include "cl.h"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/types.h>

static struct{
  PSCLContext * context;
  PSCLProgram * program;
  int           failed;
  char        * src;
}PSCLSys={NULL,NULL,0,NULL};

int PSCLUseable(){
  if(PSCLSys.failed)return 0;
  if(PSCLSys.context==NULL)return 0;
  if(PSCLSys.program==NULL)return 0;
  return 1;
}

void PSCLInit(){
  if(PSCLSys.context!=NULL)return;
  if(PSCLSys.program!=NULL)return;
  PSCLSys.context=PSCLCreate();
  
  //read src
    const char path[]="psychic.cl";
    unsigned long filesize=-1;
    struct stat statbuff;
    if(stat(path,&statbuff)<0){
      PSCLSys.failed=1;
      return;
    }else{
      filesize=statbuff.st_size;
      PSCLSys.src=(char*)malloc(filesize);
      int fd=open(path,O_RDONLY);
      if(fd==-1){
        PSCLSys.failed=1;
        return;
      }
      read(fd,PSCLSys.src,filesize);
      close(fd);
    }
  //end
  
  PSCLSys.program=PSCLCreateProgram(
    PSCLSys.context,
    PSCLSys.src
  );
}
void PSCLDestroy(){
  if(PSCLSys.context!=NULL)
    PSCLDestroyContext(PSCLSys.context);
  if(PSCLSys.program!=NULL)
    PSCLDestroyProgram(PSCLSys.program);
  PSCLSys.context=NULL;
  PSCLSys.program=NULL;
}

PSCLContext * PSCLCreate(){
  PSCLContext *ct=PSCLCreateContext();
  PSCLCreateCommandQueue(ct);
  return ct;
}

PSCLContext * PSCLCreateContext(){
  PSCLContext * ct=(PSCLContext*)malloc(sizeof(PSCLContext));
  if(ct==NULL)return NULL;
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  //选择可用的平台中的第一个
  errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if(errNum != CL_SUCCESS || numPlatforms <= 0){
    free(ct);
    return NULL;
  }
  //创建一个OpenCL上下文环境
  cl_context_properties contextProperties[] ={
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)firstPlatformId,
    0
  };
  ct->context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
    NULL, NULL, &errNum);
  return ct;
}

int PSCLCreateCommandQueue(PSCLContext * ct){
  cl_int errNum;
  cl_device_id *devices;
  size_t deviceBufferSize = -1;
  size_t mlen;
  // 获取设备缓冲区大小
  errNum = clGetContextInfo(ct->context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if (deviceBufferSize <= 0){
    return -1;
  }
  // 为设备分配缓存空间
  mlen=deviceBufferSize / sizeof(cl_device_id);
  devices = (cl_device_id*)malloc(sizeof(cl_device_id)*mlen);
  errNum = clGetContextInfo(
    ct->context,
    CL_CONTEXT_DEVICES,
    deviceBufferSize,
    devices,
    NULL
  );
  //选取可用设备中的第一个
  ct->commandQueue = clCreateCommandQueue(
    ct->context,
    devices[0],
    0,
    NULL
  );
  ct->device = devices[0];
  free(devices);
  return 0;
}

PSCLProgram * PSCLCreateProgram(PSCLContext * ct,const char* srcStr){
  
  if(ct==NULL){
    ct=PSCLSys.context;
  }
  
  PSCLProgram * pg=(PSCLProgram*)malloc(sizeof(PSCLProgram));
  if(pg==NULL)return NULL;
  pg->context=ct;
  
  cl_int errNum;
  cl_program program;
  program = clCreateProgramWithSource(
    ct->context, 1,
    (const char**)&srcStr,
    NULL,NULL
  );
  errNum = clBuildProgram(
    program, 0, NULL, NULL, NULL, NULL
  );
  pg->program=program;
  return pg;
}

PSCLKernel * PSCLCreateKernel(PSCLProgram * pg,const char * fn){
  
  if(pg==NULL){
    pg=PSCLSys.program;
  }
  
  PSCLKernel * kn=(PSCLKernel*)malloc(sizeof(PSCLKernel));
  if(kn==NULL)return NULL;
  
  kn->program=pg;
  
  kn->dim=0;
  kn->globalThreads=NULL;
  kn->localThreads=NULL;
  
  kn->args=NULL;
  kn->argn=0;
  
  kn->kernel=clCreateKernel(pg->program, fn, NULL);
  
  return kn;
}

static void PSCLMemRelease(PSCLMem * m){
  clReleaseMemObject(m->rem);
  free(m);
}

void PSCLDestroyContext(PSCLContext * ct){
  if(ct==NULL)return;
  if(ct->commandQueue != 0)
    clReleaseCommandQueue(ct->commandQueue);
  if(ct->context != 0)
    clReleaseContext(ct->context);
  free(ct);
}

void PSCLDestroyProgram(PSCLProgram * pg){
  if(pg==NULL)return;
  clReleaseProgram(pg->program);
  free(pg);
}

void PSCLDestroyKernel(PSCLKernel * kn){
  PSCLMem *arg,*barg;
  if(kn==NULL)return;
  clReleaseKernel(kn->kernel);
  if(kn->args){
    arg=kn->args;
    while(arg){
      barg=arg;
      arg=arg->next;
      PSCLMemRelease(barg);
    }
  }
  
  if(kn->globalThreads)free(kn->globalThreads);
  if(kn->localThreads) free(kn->localThreads);
  
  free(kn);
}

PSCLMem * PSCLMemAdd(PSCLKernel * kn,void * arg,size_t size,PSCLMemMethod method){
  PSCLMem * m;
  cl_mem_flags flags;
  cl_int ret;
  
  m=(PSCLMem*)malloc(sizeof(PSCLMem));
  if(m==NULL)return NULL;
  
  m->data=arg;
  m->size=size;
  m->method=method;
  m->kernel=kn;
  
  if(method==PSCLMemRead){
    flags=CL_MEM_READ_ONLY;
  }else
  if(method==PSCLMemRDWR){
    flags=CL_MEM_READ_WRITE;
  }else
  if(method==PSCLMemWrite){
    flags=CL_MEM_WRITE_ONLY;
  }
  
  //create memobject
  m->rem=clCreateBuffer(
    kn->program->context->context,
    flags,size,NULL,&ret
  );
  
  if(m->rem==NULL){
    free(m);
    return NULL;
  }
  clSetKernelArg(
    m->kernel->kernel,
    kn->argn,
    sizeof(cl_mem),&(m->rem)
  );
  //end
  
  m->next=kn->args;
  kn->args=m;
  ++(kn->argn);
  
  return m;
}

void PSCLUpdateRead(PSCLMem * m){
  if(m->method==PSCLMemWrite)return;
  clEnqueueReadBuffer(
    m->kernel->program->context->commandQueue,
    m->rem,CL_TRUE,0,m->size,m->data,0,
    NULL,NULL
  );
}

void PSCLUpdateUpload(PSCLMem * m){
  if(m->method==PSCLMemRead)return;
  clEnqueueWriteBuffer(
    m->kernel->program->context->commandQueue,
    m->rem,CL_TRUE,0,m->size,m->data,0,
    NULL,NULL
  );
}

cl_uint PSCLKernelExec(PSCLKernel * kn){
  cl_uint ret;
  if(kn->dim==00)return 0;
  if(kn->globalThreads==NULL)return 0;
  if(kn->localThreads==NULL)return 0;
  if(kn->args==NULL)return 0;
  
  ret=clEnqueueNDRangeKernel(
    kn->program->context->commandQueue,
    kn->kernel,kn->dim,NULL,
    kn->globalThreads,
    kn->localThreads,
    0,NULL,NULL
  );
  return clFinish(kn->program->context->commandQueue);
}


void PSCLKernelSetDim(PSCLKernel * kn,size_t sz){
  if(sz<=0)return;
  if(kn->globalThreads)free(kn->globalThreads);
  if(kn->localThreads) free(kn->localThreads);
  
  kn->dim=sz;
  kn->globalThreads=(size_t*)malloc(sizeof(size_t)*sz);
  kn->localThreads =(size_t*)malloc(sizeof(size_t)*sz);
}

void PSCLKernelSetGbThread(PSCLKernel * kn,const size_t * arr){
  cl_uint i;
  for(i=0;i<kn->dim;i++){
    kn->globalThreads[i]=arr[i];
  }
}

void PSCLKernelSetLCThread(PSCLKernel * kn,const size_t * arr){
  cl_uint i;
  for(i=0;i<kn->dim;i++){
    kn->localThreads[i]=arr[i];
  }
}

void PSCLKernelSetLCTAuto(PSCLKernel * kn){
  cl_uint i,tmp;
  for(i=0;i<kn->dim;i++){
    tmp=kn->globalThreads[i];
    
    if(tmp/8>4)
      kn->localThreads[i]=16;
    else
    if(tmp<8)
      kn->localThreads[i]=tmp;
    else
      kn->localThreads[i]=8;
  }
}
