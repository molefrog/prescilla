#ifndef _GL_UTILS_H_
#define _GL_UTILS_H_

#include <windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>

#include "glext.h"

extern PFNGLBINDBUFFERPROC	glBindBuffer;
extern PFNGLGENBUFFERSPROC	glGenBuffers;
extern PFNGLBUFFERDATAPROC	glBufferData;
extern PFNGLDELETEBUFFERSPROC	glDeleteBuffers;
extern PFNGLMAPBUFFERPROC	glMapBuffer;
extern PFNGLUNMAPBUFFERPROC	glUnmapBuffer;

BOOL LoadExtFunctions();

#endif //_GL_UTILS_H_

