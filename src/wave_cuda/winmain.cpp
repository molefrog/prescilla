#include <windows.h>
#include <Winternl.h>
#include <tchar.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "glutils/glutils.h"
#include "buffers.h"
#include "globals.h"

#define WND_WIDTH   1024		
#define WND_HEIGHT  768

// Название класса окна
TCHAR szClassName[] = TEXT("wndClassGL");

// Заголовок окна
TCHAR szWindowTitle[] = TEXT("Wave Equation CUDA");

BOOL needExit = FALSE;
DWORD fpsCounter = 0;

void PaintRoutine(HWND hwnd);

#define FPS_TIMER_ID		1
#define FPS_TIMER_INTERVAL	500

// Функция используется для пересчета значения FPS(frames per seconds)
void UpdateFPS(HWND hwnd) {
	TCHAR newTitle[MAX_PATH];
	
	float fps = (float) fpsCounter / ((float) FPS_TIMER_INTERVAL / 1000.0f);

	// Обновим заголовок окна
 	wsprintf(newTitle, TEXT("%s FPS %d"), szWindowTitle, (DWORD) fps);
	SetWindowText(hwnd, newTitle);

	fpsCounter  = 0;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
	switch(msg) {
		case WM_CREATE:
			// Установим таймер на просчет fps
			SetTimer(hwnd, FPS_TIMER_ID, FPS_TIMER_INTERVAL, NULL); 
		break;

		case WM_TIMER:
			if(wparam == FPS_TIMER_ID) {
				// Пересчитать и вывести новое значение fps
				UpdateFPS(hwnd);
			}
		break;

		case WM_DESTROY:
			PostQuitMessage(0);
		break;

		case WM_KEYDOWN:
			if(wparam == VK_LEFT) {
				viewAngle	+= 5.0f;
				viewAngle = viewAngle > 360.0f ? 0.0f : viewAngle;
 			}
			if(wparam == VK_RIGHT) {
				viewAngle	-= 5.0f;
				viewAngle = viewAngle > 360.0f ? 0.0f : viewAngle;
			}
			if(wparam == 0x58) {
				viewHeight	+= 10.0f;
				viewHeight = viewHeight > 360.0f ? 0.0f : viewHeight;
			}
			if(wparam == 0x5A) {
				viewHeight	-= 10.0f;
				viewHeight = viewHeight > 360.0f ? 0.0f : viewHeight;
			}
			if(wparam == VK_UP) {
				viewDistance /= 1.2f;
			}
			if(wparam == VK_DOWN) {
				viewDistance *= 1.2f;
			}
			if(wparam == VK_SPACE) pauseToggle = !pauseToggle;

		default:
			return DefWindowProc(hwnd, msg, wparam, lparam);
	}
	return 0;
}

BOOL InitGL(HWND hwnd) {
	HDC hdc = GetDC(hwnd);
	HGLRC hglrc;

	PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR),  //  size of this pfd 
		2,                     // version number 
		PFD_DRAW_TO_WINDOW |   // support window 
		PFD_SUPPORT_OPENGL |   // support OpenGL 
		PFD_DOUBLEBUFFER,      // double buffered 
		PFD_TYPE_RGBA,         // RGBA type 
		8,					   // 8-bit color depth 
		0, 0, 0, 0, 0, 0,      // color bits ignored 
		0,                     // no alpha buffer 
		0,                     // shift bit ignored 
		0,                     // no accumulation buffer 
		0, 0, 0, 0,            // accum bits ignored 
		32,                    // 32-bit z-buffer     
		0,                     // no stencil buffer 
		0,                     // no auxiliary buffer 
		PFD_MAIN_PLANE,        // main layer 
		0,                     // reserved 
		0, 0, 0                // layer masks ignored 
	};
	int pformat = ChoosePixelFormat(hdc, &pfd);
	if(!SetPixelFormat(hdc, pformat, &pfd)) {
		return FALSE;
	}

	hglrc = wglCreateContext(hdc);
	wglMakeCurrent(hdc, hglrc);

	if(!LoadExtFunctions()) {
		return FALSE;
	}

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (GLdouble) WND_WIDTH / (GLdouble) WND_HEIGHT, 0.1, 300.0);
	glMatrixMode(GL_MODELVIEW);
	glShadeModel(GL_SMOOTH);
	glViewport(0, 0, WND_WIDTH, WND_HEIGHT);

	return TRUE;
}



int WINAPI _tWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, int nShowCmd) {
	WNDCLASS	wndClass;
	HWND		hwnd;
	MSG			msg;

	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);
	UNREFERENCED_PARAMETER(nShowCmd);

	// Заполним структуру, отвечающую за класс окна
	RtlSecureZeroMemory(&wndClass, sizeof(wndClass));
	wndClass.style			= CS_OWNDC;
	wndClass.lpfnWndProc	= WindowProc;
	wndClass.hInstance		= hInstance;
	wndClass.hIcon			= LoadIcon(NULL, IDI_APPLICATION);
	wndClass.hCursor		= LoadCursor(NULL, IDC_ARROW);
	wndClass.hbrBackground	= NULL;
	wndClass.lpszMenuName	= NULL;
	wndClass.lpszClassName	= szClassName;

	// Регистрируем класс окна
	if(!RegisterClass(&wndClass)) {
		MessageBox(HWND_DESKTOP, TEXT("Cannot register window class!"), TEXT("Error!"), MB_OK);
		return 1; 
	}

	// Создаем окно
	hwnd = CreateWindow(szClassName, szWindowTitle, WS_SYSMENU,0 /*CW_USEDEFAULT*/, 0/*CW_USEDEFAULT*/, WND_WIDTH, WND_HEIGHT, HWND_DESKTOP, NULL, hInstance, 0);
	if(!hwnd) {
		MessageBox(HWND_DESKTOP, TEXT("Cannot create window!"), TEXT("Error!"), MB_OK);
		return 1;
	}	

	// Инициализируем OpenGL
	if(!InitGL(hwnd)) {
		MessageBox(HWND_DESKTOP, TEXT("OpenGL init failed!"), TEXT("Error!"), MB_OK);
		return 1;
	}

	// Важно это сделать для интеграции с OpenGL
	if(CUDA_SUCCESS != cudaGLSetGLDevice(0)) {
		return 1;
	}

	// Выделим необходимое место под буферы
	if(!InitBuffers()) {
		MessageBox(HWND_DESKTOP, TEXT("Buffers init failed!"), TEXT("Error!"), MB_OK);
		return 1;
	}

	RtlSecureZeroMemory(&msg, sizeof(MSG));
	ShowWindow(hwnd, SW_SHOW);

	// Цикл обработки оконных сообщений
	while(!needExit) {
		if(PeekMessage (&msg, NULL, 0, 0, PM_REMOVE)) {
			if(msg.message == WM_QUIT) needExit = TRUE;
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		} else {
			++fpsCounter;
			PaintRoutine(hwnd);
		}	
	}

	// Освобождаем ресурсы
	FreeBuffers();
	return 0;
}

