#include <windows.h>
#include <cuda.h>
#include <gl/GL.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "glutils/glutils.h"
#include "buffers.h"
#include "globals.h"

// Прототип функции, которая вычисляет один шаг метода по времени, 
// используя графический процессор.
void cuda_routine(void *, void *, void *, void *, unsigned int, unsigned int, float);

void CudaRoutine() {
	static float time = 0.0f;
	
	// Сдвигаем буферы сетки(настоящее теперь прошлое)
	ShiftBuffers();

	// Получим указатель(в памяти устройства) на вершинный буфер
	void * cudata_ptr;
	size_t cudata_size;
	cudaGraphicsMapResources(1, &vertexBufferResource);
	cudaGraphicsResourceGetMappedPointer(&cudata_ptr, &cudata_size, vertexBufferResource);

	if(cudata_ptr != NULL) {
		cuda_routine(layerBuffers[0], layerBuffers[1], layerBuffers[2], cudata_ptr, gridSize, threadsPerBlock, time);
		time += 0.05f;
	}
	cudaGraphicsUnmapResources(1, &vertexBufferResource);
}

void PaintRoutine(HWND hwnd) {
	// Вызываем пересчет метода
	if(!pauseToggle) CudaRoutine();

	// Очищаем буфер кадра и глубины
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearDepth(1.0);

	// Сбросим модельно-видовую матрицу
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	
	glTranslatef(0.0f, 0.0f, -viewDistance);
	glRotatef(-viewHeight, 1.0f, 0.0f, 0.0f);
	glRotatef(viewAngle, 0.0f, 0.0f, 1.0f);
	glTranslatef(-(float)gridSize*_dh/2.0f, -(float)gridSize*_dh/2.0f, 0.0f);


	glBindBuffer( GL_ARRAY_BUFFER, vertexBuffer );
	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableClientState( GL_COLOR_ARRAY );

	glVertexPointer(3, GL_FLOAT, 16, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 16, (GLvoid *)12);
	glDrawArrays(GL_POINTS, 0, gridSize*gridSize);

	glFinish();
	SwapBuffers(GetDC(hwnd));
}