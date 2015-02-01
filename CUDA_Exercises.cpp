/**Submissions for Programming assignments of Heterogeneous Parallel Programming @ Coursera.
2013 Edition.
 */

// MP 1
#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<len) out[i] = in1[i] + in2[i];
}


int main(int argc, char ** argv) {
  
  
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, \"Importing data and creating memory on host\");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, \"Importing data and creating memory on host\");

    wbLog(TRACE, \"The input length is \", inputLength);

//    wbLog(TRACE, \"The output (0,0,0) is \", (float)hostOutputImageData[0]);
  
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    wbTime_start(GPU, \"Allocating GPU memory.\");
    //@@ Allocate GPU memory here
    int size= inputLength*sizeof(float);
    cudaMalloc((void**) &deviceInput1, size);
    cudaMalloc((void**) &deviceInput2, size);
  	cudaMalloc((void**) &deviceOutput, size);
    wbTime_stop(GPU, \"Allocating GPU memory.\");

    wbTime_start(GPU, \"Copying input memory to the GPU.\");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
    wbTime_stop(GPU, \"Copying input memory to the GPU.\");
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(inputLength/1024), 1, 1);
    dim3 DimBlock(1024, 1, 1);    
      
    wbTime_start(Compute, \"Performing CUDA computation\");
    //@@ Launch the GPU Kernel here

    vecAdd<<<DimGrid,DimBlock>>>(deviceInput1,deviceInput2, deviceOutput, inputLength);
    cudaThreadSynchronize();
    wbTime_stop(Compute, \"Performing CUDA computation\");
    
    wbTime_start(Copy, \"Copying output memory to the CPU\");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, \"Copying output memory to the CPU\");

  
    wbTime_start(GPU, \"Freeing GPU Memory\");
    //@@ Free the GPU memory here
    cudaFree(deviceInput1); 
    cudaFree(deviceInput2); 
    cudaFree (deviceOutput);
    wbTime_stop(GPU, \"Freeing GPU Memory\");

    wbSolution(args, hostOutput, inputLength);
   
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}


//MP2

// MP 2: Due Sunday, Dec 16, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define wbCheck(stmt) do {                                 \\
        cudaError_t err = stmt;                            \\
        if (err != cudaSuccess) {                          \\
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \\
            return -1;                                     \\
        }                                                  \\
    } while(0)
      
# define TILE_WIDTH 16.0
 /*
      
      64: 
      The tile width  could be set at 64 based on the following reasons (which are open for discussion)
      1) The devices have a shared memory per block of 49152.
      2) Given that the kernel will use the same number of elements from 3 matrixes, this gives a memory size of 
      49152/3=16384 per matrix per block. 
      3) Given that the elements are of the type float, this means we can have 16384/sizeof(float)=4096 elements 
      per matrix per block
      =>
      4) Which gives us a tile width of square root of 4096 = 64.      
      
      32: 
      But there is a limit in the number of threads per block of 1024, which limits our tile width to square root
      of 1024=32.
      
      16:
      This tile width could be reduced for more blocks per SM. That was my decision for this program, setting it at 16.
      
      */
    
# define floatsize sizeof(float)
                  
// Compute C = A * B
__global__ void matrixMultiply(const float * A, const float * B, float * C,
			       const int numAColumns,
			       const int numCRows, const int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
      
  int CRow = blockIdx.y*TILE_WIDTH + threadIdx.y;
  int CCol = blockIdx.x*TILE_WIDTH + threadIdx.x;
  if ((CRow < numCRows) && (CCol < numCColumns)) {
    double Pvalue = 0;
    for (int i=0; i<numAColumns; i++){
        Pvalue+=(A[CRow*numAColumns+i]*B[i*numCColumns+CCol]);
     }
    C[CRow*numCColumns+CCol] = Pvalue;
    // each thread computes one element of the block sub-matrix
  }
}

int main(int argc, char ** argv) {

      wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix

  
  
    float * deviceA;
    float * deviceB;
    float * deviceC;

  
    int numARows;
	   int numAColumns; 
    int numBRows; 
  	 int numBColumns; 
    int numCRows; 
    int numCColumns; 

  
  
      int sizeA;
    int sizeB;
    int sizeC;
    //int width;
        
  	cudaError_t error;// this will be used to check for possible cuda errors
      
    args = wbArg_read(argc, argv);
  
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
      
    /*First we check if the matrix sizes for the multiplication are valid. 
    If not we show an error message and return  -1*/
    if (numAColumns!=numBRows){ 
      wbTime_stop(Generic, "Importing data and creating memory on host");
      wbLog(ERROR, "Matrix Multiplication impossible, columns of A != rows of B");
      return(-1);
    }
   
    //@@ Set numCRows and numCColumns 
    numCRows = numARows;
    numCColumns = numBColumns;
    sizeC=numCRows*numCColumns*floatsize;
  
    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(sizeC);
        
    wbTime_stop(Generic, "Importing data and creating memory on host");
  
    /*Now we establish the size of matrixes A & B, and determine width to be the largest value
    between the # of columns and the # of rows from C.*/
  
    sizeA=numARows*numAColumns*floatsize;
    sizeB=numBRows*numBColumns*floatsize;
    //if (numCRows>numCColumns){
      //width=numCRows;
    //}
    //else{
//      width=numCColumns;
  //  }

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU,"Allocating GPU memory.");
    //@@ Allocate GPU memory here
    error= cudaMalloc((void **) &deviceA, sizeA);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Allocating GPU memory.");  
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }
  
   error= cudaMalloc((void **) &deviceB, sizeB);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Allocating GPU memory.");  
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
   }
  	  
   error= cudaMalloc((void **) &deviceC, sizeC);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Allocating GPU memory.");  
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);         
    }
    
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
        
   error= cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Copying input memory to the GPU.");
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
   }
  
   error= cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Copying input memory to the GPU.");
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    //dim3 dimGrid(ceil(width/TILE_WIDTH), ceil(width/TILE_WIDTH), 1);
    dim3 dimGrid(ceil(numCColumns/TILE_WIDTH)+1, ceil(numCRows/TILE_WIDTH)+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
      
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numAColumns, numCRows, numCColumns);

    error=cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
  
    if(error != cudaSuccess){
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    error= cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
      wbTime_stop(Copy, "Copying output memory to the CPU");
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU,"Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);
                
    return 0;
  
  
  
  
                }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  






// MP3***************** MP4, MP5 below
// MP 3: Due Sunday, Dec 30, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define wbCheck(stmt) do {                                 \\
        cudaError_t err = stmt;                            \\
        if (err != cudaSuccess) {                          \\
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \\
            return -1;                                     \\
        }                                                  \\
    } while(0)

      
      
# define TILE_WIDTH 16.0
      
      /*
      
      64: 
      The tile width  could be set at 64 based on the following reasons (which are open for discussion)
      1) The devices have a shared memory per block of 49152.
      2) Given that the kernel will use the same number of elements from 3 matrixes, this gives a memory size of 
      49152/3=16384 per matrix per block. 
      3) Given that the elements are of the type float, this means we can have 16384/sizeof(float)=4096 elements 
      per matrix per block
      =>
      4) Which gives us a tile width of square root of 4096 = 64.      
      
      32: 
      But there is a limit in the number of threads per block of 1024, which limits our tile width to square root
      of 1024=32.
      
      16:
      This tile width could be reduced for more blocks per SM. That was my decision for this program, setting it at 16.
      
      */

# define floatsize sizeof(float)
      
// Compute C = A * B
__global__ void matrixMultiplyShared(const float * A, const float * B, float * C,
			             const int numCRows,const int sharedWidth, const int numCCols) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
      
    //First we create the shared memory for the tiles from A and B
    __shared__ float ds_A[(int)TILE_WIDTH][(int)TILE_WIDTH];
    __shared__ float ds_B[(int)TILE_WIDTH][(int)TILE_WIDTH];
    
    //Now we start up some variables for ease of coding...
    //This bit is similar to the example code in Lecture 4.4.
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;
    float Pvalue = 0;
      
    // Loop over the A and B tiles required to compute the C element, as in Lecture 4.4
    for (int m = 0; m<((sharedWidth - 1)/TILE_WIDTH) + 1; ++m){
    
      // Colaborative loading of A and B tiles into shared memory
      if (m>=0){
        if (Row<numCRows && m>=0 && (((m*TILE_WIDTH)+tx)<sharedWidth)){
        ds_A[ty][tx] = A[(int)(Row*sharedWidth + (m*TILE_WIDTH)+tx)];
      }
      else {
        ds_A[ty][tx]=0;
      }


      if (m*TILE_WIDTH+ty<sharedWidth && Col<numCCols){
        ds_B[ty][tx] = B[(int)((m*TILE_WIDTH+ty)*numCCols+Col)];
      }
      else {
        ds_B[ty][tx]=0;
      }
    
   
   __syncthreads(); //Now that we pass the load phase, we are ready for the use of the shared memory

   if  (Row<numCRows && Col<numCCols){
      for (int k = 0; k < TILE_WIDTH; k++)
        Pvalue += ds_A[ty][k] * ds_B[k][tx];
   }
   __syncthreads();
   
    }
      else{__syncthreads(); __syncthreads();}
    }
  
  
  if (Row<numCRows && Col<numCCols){
    C[Row*numCCols+Col] = Pvalue;
  }
}
  
      


int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)
    int sizeA;
    int sizeB;
    int sizeC;
  
    cudaError_t error;// this will be used to check for possible cuda error
    args = wbArg_read(argc, argv);
  
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    sizeC=numCRows*numCColumns*floatsize;
    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(sizeC);
        
    wbTime_stop(Generic, "Importing data and creating memory on host");
  
    /*Now we establish the size of matrixes A & B, and determine width to be the largest value
    between the # of columns and the # of rows from C.*/
  
    sizeA=numARows*numAColumns*floatsize;
    sizeB=numBRows*numBColumns*floatsize;

        

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    error= cudaMalloc((void **) &deviceA, sizeA);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Allocating GPU memory.");  
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }
  
   error= cudaMalloc((void **) &deviceB, sizeB);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Allocating GPU memory.");  
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
   }
  	  
   error= cudaMalloc((void **) &deviceC, sizeC);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Allocating GPU memory.");  
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }
    
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
   error= cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Copying input memory to the GPU.");
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
   }
  
   error= cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
      wbTime_stop(GPU, "Copying input memory to the GPU.");
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }
      
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here

   dim3 dimGrid(ceil(numCColumns/TILE_WIDTH)+1, ceil(numCRows/TILE_WIDTH)+1, 1);
   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numCColumns);

    error=cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
  
    if(error != cudaSuccess){
      wbLog(ERROR,"CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }

    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	error= cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
      wbTime_stop(Copy, "Copying output memory to the CPU");
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(error));
      return(-1);
    }
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;}



// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
// Due Tuesday, January 15, 2013 at 11:59 p.m. PST

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define floatsize sizeof(float)

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void total(float * input, float * output, int len) {
  
  __shared__ float partialSum[2*BLOCK_SIZE];
  unsigned int t=threadIdx.x;
  unsigned int start=2*blockIdx.x*blockDim.x;
  if ((start+t)<len){
    partialSum[t]=input[start+t];
  }
  else {
    partialSum[t]=0;
  }
  if ((start+t+blockDim.x)<len){ 
  partialSum[blockDim.x+t]=input[start+blockDim.x+t];
  }
  else{
    partialSum[blockDim.x+t]=0;
  }
  for (unsigned int stride= blockDim.x; stride>=1; stride/=2){
    __syncthreads();
    if (t<stride){
      partialSum[t]+=partialSum[t+stride];
    }
    
  }
  if (t==0){
    output[blockIdx.x]=partialSum[0];
  }
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * floatsize);

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**)&deviceInput, numInputElements*floatsize);
    cudaMalloc((void**)&deviceOutput, numOutputElements*floatsize);
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, numInputElements*floatsize, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here

    dim3 dimGrid(numOutputElements, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    
           
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
 
    total<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();
 
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here

    cudaMemcpy(hostOutput, deviceOutput, numOutputElements*floatsize, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here

    cudaFree(deviceInput);
    cudaFree(deviceOutput);  
  
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}



TAREA MP5*************************************************************
// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
// Due Tuesday, January 22, 2013 at 11:59 p.m. PST

#include    <wb.h>

#define BLOCK_SIZE 512.0 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void scan(float * input, float * output, int len) {
  
  __shared__ float scan_array[2*(int)BLOCK_SIZE];
  
  if((2*threadIdx.x)+(2*blockDim.x*blockIdx.x)<len){
    scan_array[2*threadIdx.x]=input[(2*threadIdx.x)+(2*blockDim.x*blockIdx.x)];
  }
  else{
     scan_array[2*threadIdx.x]=0;
  }
  if(1+(2*threadIdx.x)+(2*blockDim.x*blockIdx.x)<len){
     scan_array[1+(2*threadIdx.x)]=input[1+(2*threadIdx.x)+(2*blockDim.x*blockIdx.x)];
  }
  else{
     scan_array[1+(2*threadIdx.x)]=0;
  }
  __syncthreads();
  
  
  int stride=1;
  int index;
  while(stride<=BLOCK_SIZE){
    index=((threadIdx.x+1)*(2*stride))-1;
    if (index<2*BLOCK_SIZE && index-stride>=0){
      scan_array[index]+=scan_array[index-stride];
    }
    stride=stride*2;
    __syncthreads();
  }
  __syncthreads();
  
  for (stride= BLOCK_SIZE/2; stride>0; stride/=2){
    __syncthreads();
   index=((threadIdx.x+1)*(2*stride))-1;   
   if (index+stride<(2*BLOCK_SIZE)){
      scan_array[index+stride]+=scan_array[index];
    }  
  }
  __syncthreads();
  
  
  if((2*threadIdx.x)+(2*blockDim.x*blockIdx.x)<len){
    output[(2*threadIdx.x)+(2*blockDim.x*blockIdx.x)]=scan_array[2*threadIdx.x];
  }
  if((2*threadIdx.x)+(2*blockDim.x*blockIdx.x)<len){
     output[1+(2*threadIdx.x)+(2*blockDim.x*blockIdx.x)]=scan_array[1+(2*threadIdx.x)];
  }
  
        /*Execution order..
  
  1) Fill scan_arrays per thread. Minding the len... I think it's done.
  
  2) Reduction steps in logn. Minding the len... I think it's done.
  
  3) Post-reduction steps. Minding the len...
  
    4) Writing back to output the processed block... Minding the len...
  */
  
  
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here

    dim3 dimGrid(ceil((float)numElements/BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

  //wbLog(TRACE, "dimGrid",ceil((float)numElements/BLOCK_SIZE));
  //wbLog(TRACE, "dimBlock ",BLOCK_SIZE);
  
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce


    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);  
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

  //5) Now we need to do a post-post-reduction processing...
  
 

  
  float addElement;
  
  
  
  
  
  
  
  addElement=(float)0.00;
  
  
  
  
  
  
  
  
  
  
  for (int k=0; k<numElements;k++){
  
    
     
    
    if ((k%(int)(2*BLOCK_SIZE))==0){
      if (k!=0){
        addElement=hostOutput[k-1];
      }
    }
    hostOutput[k]+=addElement;
  }

 
  /*
  wbLog(TRACE, "Input0 ",hostInput[0]);
  wbLog(TRACE, "Input1 ",hostInput[1]);
  wbLog(TRACE, "Input2 ",hostInput[2]);
    wbLog(TRACE, "Input3 ",hostInput[3]);
  wbLog(TRACE, "Input4 ",hostInput[4]);
   wbLog(TRACE, "Input5 ",hostInput[5]);
   wbLog(TRACE, "Input6 ",hostInput[6]);
   wbLog(TRACE, "Input7 ",hostInput[7]);
wbLog(TRACE, "Input8 ",hostInput[8]);
  wbLog(TRACE, "Input9 ",hostInput[9]);
  wbLog(TRACE, "Input10 ",hostInput[10]);
  wbLog(TRACE, "Input11 ",hostInput[11]);
  
  wbLog(TRACE," Output0 ",hostOutput[0]);
  wbLog(TRACE," Output1 ",hostOutput[1]);
  wbLog(TRACE," Output2 ",hostOutput[2]);
  wbLog(TRACE," Output3 ",hostOutput[3]);
  wbLog(TRACE," Output4 ",hostOutput[4]);
  wbLog(TRACE," Output5 ",hostOutput[5]);
  wbLog(TRACE, "Output6 ",hostOutput[6]);
   wbLog(TRACE, "Output7 ",hostOutput[7]);
  wbLog(TRACE," Output8 ",hostOutput[8]);
  wbLog(TRACE," Output9 ",hostOutput[9]);
  wbLog(TRACE," Output10 ",hostOutput[10]);
  wbLog(TRACE," Output11 ",hostOutput[11]);*/
    wbSolution(args, hostOutput, numElements);
    free(hostInput);

  
   
  free(hostOutput);

    return 0;
}




*******************MP6//El codigo que sigue incluye una versión secuencial del kernel (a ser ejecutada en una sola hebra), así como modificaciones hechas al código al momento de las pruebas (notar que en el main (al lado del host) hay un doble lazo donde cambio los valores de la máscara para que sean 0 salvo en la diagonal, donde son 0,2). También imprimo en el host un montón de información innecesaria, pero que utilicé para pruebas... 
#include    <wb.h>

// Check ec2-174-129-21-232.compute-1.amazonaws.com:8080/mp/6 for more information

#define TILE_SIZE 16.0
#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE
      
      __global__ void Seq_convolution2Dkernel(const float* input, float* output, const int width, const int height, const int channels, const float* __restrict__ mask){
        if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0){    
        float Pvalue[3];
        for (int x=0; x<height; x++){ //c
        for (int y=0; y<width; y++){//c
        for(int z=0; z<channels; z++){//c
          Pvalue[z]=0;//c
     		for (int j=-2; j<3; j++){ //c
       			for (int k=-2; k<3; k++){//c
         			if ((x+j)<height && (x+j)>=0 && (y+k)<width && (y+k)>=0){//Note that mask is accessed as a constant value.
                		Pvalue[z]+=(input[((x+j)*(width)+y+k)*channels+z]*mask[(j+2)*5+k+2]);
         			}
       			}
		     }
          
          output[((x*width)+y)*channels+z]=Pvalue[z];//min(max((float)0,(float)Pvalue[z]),(float)1);
        }
        }
  		
        }}
        }
    
 __global__ void convolution2Dkernel(const float* input, float* output, const int width, const int height, const int channels, const float* __restrict__ mask){
  
   int x= (blockIdx.x*blockDim.x)+threadIdx.x;//element from input to be convoluted.
   int y= (blockIdx.y*blockDim.y)+threadIdx.y;
   
   __shared__ float shared_array[((int)TILE_SIZE+4)*((int)TILE_SIZE+4)*3];
   
   
   	  /*FIRST PHASE (OF 3)- LOADING THE SHARED ELEMENTS PER BLOCK, FROM INPUT*/
   		int ty=threadIdx.y;
   		int tx=threadIdx.x;
   
  	

   		
   		float Pvalue[3];
   		for (int l=0; l<channels; l++){
     		Pvalue[l]=(float)0;
     		//Now we're loading a regular element
          if (x<height&&y<width&& x>=0 && y>=0){
            shared_array[((tx+2)*((int)TILE_SIZE+4)+ty+2)*channels+l]=input[((x*width)+y)*channels+l];}
          else{
            shared_array[((tx+2)*((int)TILE_SIZE+4)+ty+2)*channels+l]=0;
          }
         __syncthreads(); 
     
          //NOTE: My solution for loading boundary and corner halo elements might be un-elegant, but I think it's useful for understanding the cases in a simple manner.
     	//Here we're going to check if the element is a corner or an element lying at the boundaries of the tile.
     
          	int corner=-1;
           	int boundary=-1;
     	    int corner_add_x[8]; //We're going to store in this array the offsets of the 8 extra elements we have to load (from each corner) for the halo of the tile.
     		int corner_add_y[8];
     		int boundary_add_x[2]; //If we are at a boundary, we'll only need to load 2 extra elements for the halo of the tile. 
     		int boundary_add_y[2];
     		if(tx==0 && ty==0){//corners
	     			corner=0;
       				corner_add_x[0]=-1; corner_add_y[0]=-1;
       				corner_add_x[1]=-2; corner_add_y[1]=-1;
       				corner_add_x[2]=-1; corner_add_y[2]=-2;
       				corner_add_x[3]=-2; corner_add_y[3]=-2;
       				corner_add_x[4]=0; corner_add_y[4]=-1;
	   				corner_add_x[5]=0; corner_add_y[5]=-2;
	   				corner_add_x[6]=-1; corner_add_y[6]=0;
	   				corner_add_x[7]=-2; corner_add_y[7]=0;
     		}
     		else if(tx==0 && ty==TILE_SIZE-1){
     			    corner=0;
       				corner_add_x[0]=-1; corner_add_y[0]=1;
       				corner_add_x[1]=-2; corner_add_y[1]=1;
       				corner_add_x[2]=-1; corner_add_y[2]=2;
       				corner_add_x[3]=-2; corner_add_y[3]=2;
       				corner_add_x[4]=0; corner_add_y[4]=1;
	   				corner_add_x[5]=0; corner_add_y[5]=2;
	   				corner_add_x[6]=-1; corner_add_y[6]=0;
	   				corner_add_x[7]=-2; corner_add_y[7]=0;
     		}
     		else if(tx==TILE_SIZE-1 && ty==0){
       				corner=0;
       				corner_add_x[0]=1; corner_add_y[0]=-1;
       				corner_add_x[1]=2; corner_add_y[1]=-1;
       				corner_add_x[2]=1; corner_add_y[2]=-2;
       				corner_add_x[3]=2; corner_add_y[3]=-2;
       				corner_add_x[4]=0; corner_add_y[4]=-1;
	   				corner_add_x[5]=0; corner_add_y[5]=-2;
	   				corner_add_x[6]=1; corner_add_y[6]=0;
	   				corner_add_x[7]=2; corner_add_y[7]=0;
     		}
     		else if (tx==TILE_SIZE-1 && ty==TILE_SIZE-1){
       				corner=0;
       				corner_add_x[0]=1; corner_add_y[0]=1;
       				corner_add_x[1]=2; corner_add_y[1]=1;
       				corner_add_x[2]=1; corner_add_y[2]=2;
       				corner_add_x[3]=2; corner_add_y[3]=2;
       				corner_add_x[4]=0; corner_add_y[4]=1;
	   				corner_add_x[5]=0; corner_add_y[5]=2;
	   				corner_add_x[6]=1; corner_add_y[6]=0;
	   				corner_add_x[7]=2; corner_add_y[7]=0;
     		}
     		else if(tx==0 && (ty!=0 && ty!=TILE_SIZE-1)){ //boundaries
       				boundary=0;
       				boundary_add_x[0]=-1; boundary_add_y[0]=0;
       				boundary_add_x[1]=-2; boundary_add_y[1]=0;
     		}
     		else if (ty==0 && (tx!=0 && tx!=TILE_SIZE-1)){
       				boundary=0;
       				boundary_add_x[0]=0; boundary_add_y[0]=-1;
       				boundary_add_x[1]=0; boundary_add_y[1]=-2;
     		}
     		else if(tx==TILE_SIZE-1 && (ty!=0 && ty!=TILE_SIZE-1) ){
       				boundary=0;
       				boundary_add_x[0]=1; boundary_add_y[0]=0;
       				boundary_add_x[1]=2; boundary_add_y[1]=0;
     		}
     		else if (ty==TILE_SIZE-1 && (tx!=0 && tx!=TILE_SIZE-1)){
       				boundary=0;
       				boundary_add_x[0]=0; boundary_add_y[0]=1;
       				boundary_add_x[1]=0; boundary_add_y[1]=2;
     		}
     		else{}
     
            if (corner==0){//We'll only load valid elements from input (no setting of dummy elements to 0). 
			       for (int i=0; i<8; i++){
         				if(x+corner_add_x[i]>=0 && x+corner_add_x[i]<height && y+corner_add_y[i]>=0 && y+corner_add_y[i]<width){
           					shared_array[((tx+2+corner_add_x[i])*((int)TILE_SIZE+4)+ty+2+corner_add_y[i])*channels+l]=input[((x+corner_add_x[i])*width+y+corner_add_y[i])*channels+l];
         				}
                     	else{
                          	shared_array[((tx+2+corner_add_x[i])*((int)TILE_SIZE+4)+ty+2+corner_add_y[i])*channels+l]=0;
                     	}
       				}
     		}
     		if (boundary==0){
     			  for (int i=0; i<2; i++){
         				if(x+boundary_add_x[i]>=0 && x+boundary_add_x[i]<height && y+boundary_add_y[i]>=0 && y+boundary_add_y[i]<width){
  					        shared_array[((tx+2+boundary_add_x[i])*((int)TILE_SIZE+4)+ty+2+boundary_add_y[i])*channels+l]=input[((x+boundary_add_x[i])*width+y+boundary_add_y[i])*channels+l];
         				}
                    	else{
                            shared_array[((tx+2+boundary_add_x[i])*((int)TILE_SIZE+4)+ty+2+boundary_add_y[i])*channels+l]=0;
                    	}
       				}
     		}
          	else{}
     
      		__syncthreads(); 
   		}
   		__syncthreads(); //END OF FIRST PHASE. THE ELEMENTS SHOULD BE LOADED ON THE SHARED_ARRAY.
     
   		//SECOND PHASE- APPLYING THE CONVOLUTION ON THE SHARED DATA AND STORING ON A LOCAL ARRAY.

   		for(int z=0; z<channels; z++){
     		for (int j=-2; j<3; j++){
       			for (int k=-2; k<3; k++){
         			if ((x+j)<height && (x+j)>=0 && (y+k)<width && (y+k)>=0){//Note that mask is accessed as a constant value.
                		Pvalue[z]+=(shared_array[((tx+2+j)*((int)TILE_SIZE+4)+ty+2+k)*channels+z]*mask[(j+2)*5+k+2]);
         			}
       			}
		     }
   		}
  		__syncthreads();//This barrier is not really necessary, but I think it might help platform optimizations to improve the performance, by separating the phases of the execution according to  memory use. 
 	
     	//THIRD PHASE- WRITING THE RESULTS TO OUTPUT.
     
  		 if (x<height && y<width && x>=0 && y>=0){
           	for (int z=0; z<channels; z++){
     	   		 output[((x*width)+y)*channels+z]=min(max((float)0,(float)Pvalue[z]),(float)1);
   			}	
   		}
    
 }



int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


  for (int it=0; it<5; it++){
    for (int it2=0; it2<5; it2++){
      if (it2==it){
        hostMaskData[it*5+it2]=(float)0.2;
      }
      else{
        hostMaskData[it*5+it2]=0;
      }
    }
  }

   wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    
    dim3 dimGrid((int)ceil(imageHeight/TILE_SIZE)+1, (int)ceil(imageWidth/TILE_SIZE)+1, 1);
     dim3 dimBlock((int)TILE_SIZE, (int)TILE_SIZE, 1);
     //dim3 dimGrid (1,1,1);
     //dim3 dimBlock(1, 1, 1);

  
  
    convolution2Dkernel<<<dimGrid, dimBlock>>>((const float*)deviceInputImageData,(float*)deviceOutputImageData, imageWidth, imageHeight, imageChannels, (const float*)deviceMaskData);

//    Seq_convolution2Dkernel<<<dimGrid, dimBlock>>>((const float*)deviceInputImageData,(float*)deviceOutputImageData, imageWidth, imageHeight, imageChannels, (const float*)deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

   
  wbLog(TRACE,"Processed image width= ", imageWidth);
  wbLog(TRACE,"Processed image height= ", imageHeight);
  wbLog(TRACE,"Processed image channels= ", imageChannels);
  cudaError_t err2 = cudaGetLastError(); wbLog(TRACE, "Error =",(int) err2, cudaGetErrorString(err2));
          
    
  wbLog(TRACE,"input 0,47 ", hostInputImageData[47*3+0]);
    wbLog(TRACE,"output 0,47 ", hostOutputImageData[47*3+0]);
    wbLog(TRACE,"input 1,47 ", hostInputImageData[((1*imageWidth)+47)*3+0]);
    wbLog(TRACE,"output 1,47 ", hostOutputImageData[((1*imageWidth)+47)*3+0]);
  wbLog(TRACE,"input 2,47 ", hostInputImageData[((2*imageWidth)+47)*3+0]);
    wbLog(TRACE,"output 2,47 ", hostOutputImageData[((2*imageWidth)+47)*3+0]);
          wbLog(TRACE,"input 0,48 ", hostInputImageData[48*3+0]);
    wbLog(TRACE,"output 0,48 ", hostOutputImageData[48*3+0]);
    wbLog(TRACE,"input 1,48 ", hostInputImageData[((1*imageWidth)+48)*3+0]);
    wbLog(TRACE,"output 1,48 ", hostOutputImageData[((1*imageWidth)+48)*3+0]);
  wbLog(TRACE,"input 2,48 ", hostInputImageData[((2*imageWidth)+48)*3+0]);
    wbLog(TRACE,"output 2,48 ", hostOutputImageData[((2*imageWidth)+48)*3+0]);
          wbLog(TRACE,"input 0,48 ", hostInputImageData[48*3+0]);
    wbLog(TRACE,"output 0,48 ", hostOutputImageData[48*3+0]);
    wbLog(TRACE,"input 1,48 ", hostInputImageData[((1*imageWidth)+48)*3+0]);
    wbLog(TRACE,"output 1,48 ", hostOutputImageData[((1*imageWidth)+48)*3+0]);
  wbLog(TRACE,"input 2,48 ", hostInputImageData[((2*imageWidth)+48)*3+0]);
    wbLog(TRACE,"output 2,48 ", hostOutputImageData[((2*imageWidth)+48)*3+0]);
              wbLog(TRACE,"input 0,49 ", hostInputImageData[49*3+0]);
    wbLog(TRACE,"output 0,49 ", hostOutputImageData[49*3+0]);
    wbLog(TRACE,"input 1,49 ", hostInputImageData[((1*imageWidth)+49)*3+0]);
    wbLog(TRACE,"output 1,49 ", hostOutputImageData[((1*imageWidth)+49)*3+0]);
  wbLog(TRACE,"input 2,49 ", hostInputImageData[((2*imageWidth)+49)*3+0]);
    wbLog(TRACE,"output 2,49 ", hostOutputImageData[((2*imageWidth)+49)*3+0]);
    wbLog(TRACE,"input 0,50 ", hostInputImageData[50*3+0]);
    wbLog(TRACE,"output 0,50 ", hostOutputImageData[50*3+0]);
    wbLog(TRACE,"input 1,50 ", hostInputImageData[((1*imageWidth)+50)*3+0]);
    wbLog(TRACE,"output 1,50 ", hostOutputImageData[((1*imageWidth)+50)*3+0]);
  wbLog(TRACE,"input 2,50 ", hostInputImageData[((2*imageWidth)+50)*3+0]);
    wbLog(TRACE,"output 2,50 ", hostOutputImageData[((2*imageWidth)+50)*3+0]);      
    wbLog(TRACE,"input 0,51 ", hostInputImageData[51*3+0]);
    wbLog(TRACE,"output 0,51 ", hostOutputImageData[51*3+0]);
    wbLog(TRACE,"input 1,51 ", hostInputImageData[((1*imageWidth)+51)*3+0]);
    wbLog(TRACE,"output 1,51 ", hostOutputImageData[((1*imageWidth)+51)*3+0]);
  wbLog(TRACE,"input 2,51 ", hostInputImageData[((2*imageWidth)+51)*3+0]);
    wbLog(TRACE,"output 2,51 ", hostOutputImageData[((2*imageWidth)+51)*3+0]);
    wbLog(TRACE,"mask 0,0 ", hostMaskData[0*5+0]);
    wbLog(TRACE,"mask 0,1 ", hostMaskData[0*5+1]);
    wbLog(TRACE,"mask 0,2  ", hostMaskData[0*5+2]);
    wbLog(TRACE,"mask 0,3  ", hostMaskData[0*5+3]);
    wbLog(TRACE,"mask 0,4  ", hostMaskData[0*5+4]);
    wbLog(TRACE,"mask 1,0  ", hostMaskData[1*5+0]);
    wbLog(TRACE,"mask 1,1 ", hostMaskData[1*5+1]);
    wbLog(TRACE,"mask 1,2  ", hostMaskData[1*5+2]);
    wbLog(TRACE,"mask 1,3  ", hostMaskData[1*5+3]);
    wbLog(TRACE,"mask 1,4  ", hostMaskData[1*5+4]);
    wbLog(TRACE,"mask 2,0  ", hostMaskData[2*5+0]);
    wbLog(TRACE,"mask 2,1 ", hostMaskData[2*5+1]);
    wbLog(TRACE,"mask 2,2  ", hostMaskData[2*5+2]);
    wbLog(TRACE,"mask 2,3  ", hostMaskData[2*5+3]);
    wbLog(TRACE,"mask 2,4  ", hostMaskData[2*5+4]);
    wbLog(TRACE,"mask 3,0  ", hostMaskData[3*5+0]);
    wbLog(TRACE,"mask 3,1 ", hostMaskData[3*5+1]);
    wbLog(TRACE,"mask 3,2  ", hostMaskData[3*5+2]);
    wbLog(TRACE,"mask 3,3  ", hostMaskData[3*5+3]);
    wbLog(TRACE,"mask 3,4  ", hostMaskData[3*5+4]);
    wbLog(TRACE,"mask 4,0  ", hostMaskData[4*5+0]);
    wbLog(TRACE,"mask 4,1 ", hostMaskData[4*5+1]);
    wbLog(TRACE,"mask 4,2  ", hostMaskData[4*5+2]);
    wbLog(TRACE,"mask 4,3  ", hostMaskData[4*5+3]);
    wbLog(TRACE,"mask 4,4  ", hostMaskData[4*5+4]);
          
    wbSolution(arg, outputImage);        
  
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    
  wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}



//MP6

#include    <wb.h>

// Check ec2-174-129-21-232.compute-1.amazonaws.com:8080/mp/6 for more information

#define TILE_SIZE 16.0
#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE
      
      __global__ void Seq_convolution2Dkernel(const float* input, float* output, const int width, const int height, const int channels, const float* __restrict__ mask){
        if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0){    
        float Pvalue[3];
        for (int x=0; x<height; x++){ //c
        for (int y=0; y<width; y++){//c
        for(int z=0; z<channels; z++){//c
          Pvalue[z]=0;//c
     		for (int j=-2; j<3; j++){ //c
       			for (int k=-2; k<3; k++){//c
         			if ((x+j)<height && (x+j)>=0 && (y+k)<width && (y+k)>=0){//Note that mask is accessed as a constant value.
                		Pvalue[z]+=(input[((x+j)*(width)+y+k)*channels+z]*mask[(j+2)*5+k+2]);
         			}
       			}
		     }
          
          output[((x*width)+y)*channels+z]=Pvalue[z];//min(max((float)0,(float)Pvalue[z]),(float)1);
        }
        }
  		
        }}
        }
    
 __global__ void convolution2Dkernel(const float* input, float* output, const int width, const int height, const int channels, const float* __restrict__ mask){
  
   int x= (blockIdx.x*blockDim.x)+threadIdx.x;//element from input to be convoluted.
   int y= (blockIdx.y*blockDim.y)+threadIdx.y;
   
   __shared__ float shared_array[((int)TILE_SIZE+4)*((int)TILE_SIZE+4)*3];
   
   
   	  /*FIRST PHASE (OF 3)- LOADING THE SHARED ELEMENTS PER BLOCK, FROM INPUT*/
   		int ty=threadIdx.y;
   		int tx=threadIdx.x;	

   		
   		float Pvalue[3];
   		for (int l=0; l<channels; l++){
     		Pvalue[l]=(float)0;
     		//Now we're loading a regular element
          if (x<height&&y<width&& x>=0 && y>=0){
            shared_array[((tx+2)*((int)TILE_SIZE+4)+ty+2)*channels+l]=input[((x*width)+y)*channels+l];}
          else{
            shared_array[((tx+2)*((int)TILE_SIZE+4)+ty+2)*channels+l]=0;
          }
         __syncthreads(); 
     
          //NOTE: My solution for loading boundary and corner halo elements might be un-elegant, but I think it's useful for understanding the cases in a simple manner.
     	//Here we're going to check if the element is a corner or an element lying at the boundaries of the tile.
     
          	int corner=-1;
           	int boundary=-1;
     	    int corner_add_x[8]; //We're going to store in this array the offsets of the 8 extra elements we have to load (from each corner) for the halo of the tile.
     		int corner_add_y[8];
     		int boundary_add_x[2]; //If we are at a boundary, we'll only need to load 2 extra elements for the halo of the tile. 
     		int boundary_add_y[2];
     		if(tx==0 && ty==0){//corners
	     			corner=0;
       				corner_add_x[0]=-1; corner_add_y[0]=-1;
       				corner_add_x[1]=-2; corner_add_y[1]=-1;
       				corner_add_x[2]=-1; corner_add_y[2]=-2;
       				corner_add_x[3]=-2; corner_add_y[3]=-2;
       				corner_add_x[4]=0; corner_add_y[4]=-1;
	   				corner_add_x[5]=0; corner_add_y[5]=-2;
	   				corner_add_x[6]=-1; corner_add_y[6]=0;
	   				corner_add_x[7]=-2; corner_add_y[7]=0;
     		}
     		else if(tx==0 && ty==TILE_SIZE-1){
     			    corner=0;
       				corner_add_x[0]=-1; corner_add_y[0]=1;
       				corner_add_x[1]=-2; corner_add_y[1]=1;
       				corner_add_x[2]=-1; corner_add_y[2]=2;
       				corner_add_x[3]=-2; corner_add_y[3]=2;
       				corner_add_x[4]=0; corner_add_y[4]=1;
	   				corner_add_x[5]=0; corner_add_y[5]=2;
	   				corner_add_x[6]=-1; corner_add_y[6]=0;
	   				corner_add_x[7]=-2; corner_add_y[7]=0;
     		}
     		else if(tx==TILE_SIZE-1 && ty==0){
       				corner=0;
       				corner_add_x[0]=1; corner_add_y[0]=-1;
       				corner_add_x[1]=2; corner_add_y[1]=-1;
       				corner_add_x[2]=1; corner_add_y[2]=-2;
       				corner_add_x[3]=2; corner_add_y[3]=-2;
       				corner_add_x[4]=0; corner_add_y[4]=-1;
	   				corner_add_x[5]=0; corner_add_y[5]=-2;
	   				corner_add_x[6]=1; corner_add_y[6]=0;
	   				corner_add_x[7]=2; corner_add_y[7]=0;
     		}
     		else if (tx==TILE_SIZE-1 && ty==TILE_SIZE-1){
       				corner=0;
       				corner_add_x[0]=1; corner_add_y[0]=1;
       				corner_add_x[1]=2; corner_add_y[1]=1;
       				corner_add_x[2]=1; corner_add_y[2]=2;
       				corner_add_x[3]=2; corner_add_y[3]=2;
       				corner_add_x[4]=0; corner_add_y[4]=1;
	   				corner_add_x[5]=0; corner_add_y[5]=2;
	   				corner_add_x[6]=1; corner_add_y[6]=0;
	   				corner_add_x[7]=2; corner_add_y[7]=0;
     		}
     		else if(tx==0 && (ty!=0 && ty!=TILE_SIZE-1)){ //boundaries
       				boundary=0;
       				boundary_add_x[0]=-1; boundary_add_y[0]=0;
       				boundary_add_x[1]=-2; boundary_add_y[1]=0;
     		}
     		else if (ty==0 && (tx!=0 && tx!=TILE_SIZE-1)){
       				boundary=0;
       				boundary_add_x[0]=0; boundary_add_y[0]=-1;
       				boundary_add_x[1]=0; boundary_add_y[1]=-2;
     		}
     		else if(tx==TILE_SIZE-1 && (ty!=0 && ty!=TILE_SIZE-1) ){
       				boundary=0;
       				boundary_add_x[0]=1; boundary_add_y[0]=0;
       				boundary_add_x[1]=2; boundary_add_y[1]=0;
     		}
     		else if (ty==TILE_SIZE-1 && (tx!=0 && tx!=TILE_SIZE-1)){
       				boundary=0;
       				boundary_add_x[0]=0; boundary_add_y[0]=1;
       				boundary_add_x[1]=0; boundary_add_y[1]=2;
     		}
     		else{}
     
            if (corner==0){//We'll only load valid elements from input (no setting of dummy elements to 0). 
			       for (int i=0; i<8; i++){
         				if(x+corner_add_x[i]>=0 && x+corner_add_x[i]<height && y+corner_add_y[i]>=0 && y+corner_add_y[i]<width){
           					shared_array[((tx+2+corner_add_x[i])*((int)TILE_SIZE+4)+ty+2+corner_add_y[i])*channels+l]=input[((x+corner_add_x[i])*width+y+corner_add_y[i])*channels+l];
         				}
                     	else{
                          	shared_array[((tx+2+corner_add_x[i])*((int)TILE_SIZE+4)+ty+2+corner_add_y[i])*channels+l]=0;
                     	}
       				}
     		}
     		if (boundary==0){
     			  for (int i=0; i<2; i++){
         				if(x+boundary_add_x[i]>=0 && x+boundary_add_x[i]<height && y+boundary_add_y[i]>=0 && y+boundary_add_y[i]<width){
  					        shared_array[((tx+2+boundary_add_x[i])*((int)TILE_SIZE+4)+ty+2+boundary_add_y[i])*channels+l]=input[((x+boundary_add_x[i])*width+y+boundary_add_y[i])*channels+l];
         				}
                    	else{
                            shared_array[((tx+2+boundary_add_x[i])*((int)TILE_SIZE+4)+ty+2+boundary_add_y[i])*channels+l]=0;
                    	}
       				}
     		}
          	else{}
     
      		__syncthreads(); 
   		}
   		__syncthreads(); //END OF FIRST PHASE. THE ELEMENTS SHOULD BE LOADED ON THE SHARED_ARRAY.
     
   		//SECOND PHASE- APPLYING THE CONVOLUTION ON THE SHARED DATA AND STORING ON A LOCAL ARRAY.

   		for(int z=0; z<channels; z++){
     		for (int j=-2; j<3; j++){
       			for (int k=-2; k<3; k++){
         			if ((x+j)<height && (x+j)>=0 && (y+k)<width && (y+k)>=0){//Note that mask is accessed as a constant value.
                		Pvalue[z]+=(shared_array[((tx+2+j)*((int)TILE_SIZE+4)+ty+2+k)*channels+z]*mask[(j+2)*5+k+2]);
         			}
       			}
		     }
   		}
  		__syncthreads();//This barrier is not really necessary, but I think it might help platform optimizations to improve the performance, by separating the phases of the execution according to  memory use. 
 	
     	//THIRD PHASE- WRITING THE RESULTS TO OUTPUT.
     
  		 if (x<height && y<width && x>=0 && y>=0){
           	for (int z=0; z<channels; z++){
     	   		 output[((x*width)+y)*channels+z]=min(max((float)0,(float)Pvalue[z]),(float)1);
   			}	
   		}
    
 }



int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

   wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    
    dim3 dimGrid((int)ceil(imageHeight/TILE_SIZE)+1, (int)ceil(imageWidth/TILE_SIZE)+1, 1);
     dim3 dimBlock((int)TILE_SIZE, (int)TILE_SIZE, 1);
     //dim3 dimGrid (1,1,1);
     //dim3 dimBlock(1, 1, 1);

  
  
    convolution2Dkernel<<<dimGrid, dimBlock>>>((const float*)deviceInputImageData,(float*)deviceOutputImageData, imageWidth, imageHeight, imageChannels, (const float*)deviceMaskData);

//    Seq_convolution2Dkernel<<<dimGrid, dimBlock>>>((const float*)deviceInputImageData,(float*)deviceOutputImageData, imageWidth, imageHeight, imageChannels, (const float*)deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

   
  wbLog(TRACE,"Processed image width= ", imageWidth);
  wbLog(TRACE,"Processed image height= ", imageHeight);
  wbLog(TRACE,"Processed image channels= ", imageChannels);
  cudaError_t err2 = cudaGetLastError(); wbLog(TRACE, "Error =",(int) err2, cudaGetErrorString(err2));
          
    wbSolution(arg, outputImage);        
  
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    
  wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

