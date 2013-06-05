#ifndef __COMMON_H__
#define __COMMON_H__

#include <vector>
#include <string>
#include <sys/time.h>

#include "../config.h"

#define pdMAX(a,b)  ((a)>(b))?(a):(b)
#define pdMIN(a,b)  ((a)<(b))?(a):(b)

#define tol 0.5f

#ifdef FINE_DEBUG
#define CHECK_MATCH(A,B,C) { if((A>tol+B) || (A<B-tol)){ \
      printf("ERROR: %s %f != %f\n", C, A, B);}}
#else
#define CHECK_MATCH(A,B,C)
#endif

//helper function with parsing lines
void StringSplit(std::vector<std::string> *results, std::string str, std::string delim);

// FINE CPU timer
// FIXME!! Don't declare as class member object
// (will occur segmentation fault)
class Timer {
public:

	Timer( void );

	void  Start      ( void );
	void  Stop       ( void );
	float ElapsedTime( void );	// milliseconds

	~Timer( void );

private:
	struct timespec _timeStart;
	struct timespec _timeStop;
};

#endif
