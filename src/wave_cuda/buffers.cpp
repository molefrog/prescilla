#include <windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "glutils/glutils.h"
#include "buffers.h"
#include "globals.h"

GLuint vertexBuffer;
struct cudaGraphicsResource * vertexBufferResource;

// Массив буферов для сетки
void * layerBuffers[NUM_BUFFERS];

// Сдвигает буферы вправо
void ShiftBuffers() {
	// Запоминаем последний буфер
	void * last = layerBuffers[NUM_BUFFERS - 1];
	
	// Сдвигаем с конца вправо
	for(int i = NUM_BUFFERS-1; i > 0; --i) {
		layerBuffers[i] = layerBuffers[i-1];
	}
	// На место первого ставим последний
	layerBuffers[0] = last;
}

// Инициализирует буферы
BOOL InitBuffers() {
	// Выделяем память под буферы сетки и заполняем нулями
	for(int i=0; i<NUM_BUFFERS; ++i) {
		cudaError_t err = cudaMalloc(&(layerBuffers[i]), gridSize * gridSize * sizeof(float));
		err = cudaMemset(layerBuffers[i], 0, gridSize*gridSize*sizeof(float));

		int a = 2;
	}

	// Создаем вершинный буфер OpenGL и региструем его в CUDA
	glGenBuffers( 1, &vertexBuffer);
	glBindBuffer( GL_ARRAY_BUFFER, vertexBuffer); 
	glBufferData( GL_ARRAY_BUFFER, gridSize * gridSize * sizeof(buffer_elem), NULL, GL_DYNAMIC_COPY );

	// Заполняем вершинный буфер нулями
	void * buf = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	memset(buf, 0, gridSize * gridSize * sizeof(buffer_elem));
	glUnmapBuffer(GL_ARRAY_BUFFER);

	if(CUDA_SUCCESS != cudaGraphicsGLRegisterBuffer(&(vertexBufferResource), vertexBuffer, cudaGraphicsMapFlagsNone)) {
		return FALSE;
	}

	return TRUE;
}

// Освобождает память, выделенную под буферы
BOOL FreeBuffers() {
	for(int i=0; i<NUM_BUFFERS; ++i) {
		cudaFree(layerBuffers[i]);
	}
	
	// Убираем вершнинный буфер из CUDA
	cudaGraphicsUnregisterResource(vertexBufferResource);
	glDeleteBuffers(1,  &vertexBuffer);
		
	return TRUE;
}