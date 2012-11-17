#ifndef _GLOBALS_H_
#define _GLOBALS_H_

// Параметры задачи
// TODO: вынесение этих значений для их изменения без перекомпиляции!
// Основная проблема в том, что они используются графическими процессорами,
// а изменяются CPU. Глобальная память разная!
#define _dh			(0.1f)		// Шаг по сетке
#define _dt			(0.05f)		// Шаг по времени
#define _friction	(1.0f)		// Коэффициент трения
#define _speed		(1.0f)		// Скорость распространения волны

extern unsigned int threadsPerBlock;
extern unsigned int gridSize;

extern float viewAngle;
extern float viewHeight;
extern float viewDistance;

extern bool	 pauseToggle;

#endif //_GLOBALS_H_