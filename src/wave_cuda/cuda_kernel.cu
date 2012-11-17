#include <windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>

#include "buffers.h"
#include "globals.h"

__device__ float calc_forces(float x, float y, float time) {
	if( x < 0.55f && x > 0.45f && y < 0.55f && y > 0.45f ) return 20.0f*sin(time*2.0f);

	return 0.0; 10.0f*sin(3.0f*time*x)*cos(10.0f*time*y);
}

__global__ void cuda_kernel(void * g0, void * g1, void * g2, void * vert, unsigned int GridDim, float time) {
	// Вычислим координаты текущего CUDA-потока
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	#define _G0(X, Y) (((float *) g0) [(Y)*GridDim + (X)])
	#define _G1(X, Y) (((float *) g1) [(Y)*GridDim + (X)])
	#define _G2(X, Y) (((float *) g2) [(Y)*GridDim + (X)])
	#define  _V(X, Y) (((buffer_elem *) vert) [(Y)*GridDim + (X)])

	// Проверка, что точка не граничная!
	if(x < GridDim-1 && y < GridDim-1 && x > 0 && y > 0) {
		// Считаем скорость(первая производная по времени)
		float velocity = (_G1(x,y) - _G2(x,y)) / _dt;

		// Считаем градиент в момент времени t-1
		float grad = (_G1(x-1, y) + _G1(x+1,y) + _G1(x, y-1) + _G1(x,y+1)- 4.0f*_G1(x,y)) / (_dh * _dh);
	
		// Вычислим внешние силы (без учета трения)
		float forces = calc_forces((float)x/(float)GridDim, (float)y/(float)GridDim, time);

		// Считаем окончательное значение в точке
		_G0(x, y) = _speed * _speed * _dt * _dt * (grad + forces - _friction * velocity)  + 2.0f*_G1(x,y) - _G2(x,y);
	}

	_V(x, y).x = (GLfloat) x * _dh;
	_V(x, y).y = (GLfloat) y * _dh;
	_V(x, y).z = _G0(x, y);
	_V(x, y).r = (unsigned int)(_G0(x, y)*10000.0f)%256;
	_V(x, y).g = 70 + (unsigned int)(_G0(x, y)*100.0f)%256;
	_V(x, y).b = (unsigned int)(_G0(x, y)*1000.0f)%256;
}


void cuda_routine(void * g0, void * g1, void * g2, void * vert, unsigned int GridDim, unsigned int threads, float time) {
	dim3 blockDim(GridDim / threads, GridDim / threads);
	dim3 threadsDim(threads, threads);

	cuda_kernel<<< blockDim, threadsDim >>> (g0, g1, g2, vert, GridDim, time);
}



