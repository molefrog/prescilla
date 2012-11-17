#include "glutils.h"

PFNGLBINDBUFFERPROC	glBindBuffer;
PFNGLGENBUFFERSPROC	glGenBuffers;
PFNGLBUFFERDATAPROC	glBufferData;
PFNGLDELETEBUFFERSPROC	glDeleteBuffers;
PFNGLMAPBUFFERPROC	glMapBuffer;
PFNGLUNMAPBUFFERPROC	glUnmapBuffer;

BOOL LoadExtFunctions() {
	glBindBuffer = (PFNGLBINDBUFFERPROC) wglGetProcAddress("glBindBuffer");
	if(glBindBuffer == NULL) return FALSE;

	glGenBuffers = (PFNGLGENBUFFERSPROC) wglGetProcAddress("glGenBuffers");
	if(glGenBuffers == NULL) return FALSE;

	glBufferData = (PFNGLBUFFERDATAPROC) wglGetProcAddress("glBufferData");
	if(glBufferData == NULL) return FALSE;

	glDeleteBuffers = (PFNGLDELETEBUFFERSPROC) wglGetProcAddress("glDeleteBuffers");
	if(glDeleteBuffers == NULL) return FALSE;

	glMapBuffer = (PFNGLMAPBUFFERPROC) wglGetProcAddress("glMapBuffer");
	if(glMapBuffer == NULL) return FALSE;

	glUnmapBuffer = (PFNGLUNMAPBUFFERPROC) wglGetProcAddress("glUnmapBuffer");
	if(glUnmapBuffer == NULL) return FALSE;

	return TRUE;
}