#ifndef _BUFFERS_H_
#define _BUFFERS_H_

#include <windows.h>
#include <gl/gl.h>

#define NUM_BUFFERS 3

#if (NUM_BUFFERS < 3)
#error "Wrong number of buffers!"
#endif

extern void * layerBuffers[NUM_BUFFERS];

typedef struct _buffer_elem {
	GLfloat x, y, z;
	GLubyte r, g, b, a;
} buffer_elem;

extern unsigned int GridDim;

extern GLuint vertexBuffer;
extern struct cudaGraphicsResource * vertexBufferResource;

void ShiftBuffers();
BOOL InitBuffers();
BOOL FreeBuffers();

#endif //_BUFFERS_H_