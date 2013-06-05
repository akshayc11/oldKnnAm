#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <sys/time.h>

#include "common.h"

using namespace std;

void StringSplit(vector<string> *results, string str, string delim) {
  int cutAt = str.find_first_of(delim);
  while( (cutAt >=0) && (cutAt < str.npos)){
    if(cutAt > 0){
      results->push_back(str.substr(0, cutAt));
    }
    str = str.substr(cutAt+1);
    cutAt = str.find_first_of(delim);
  }
  if(str.length() > 0){
    results->push_back(str);
  }

  return;
}

// Timer class
Timer::Timer ( void ) {
}

Timer::~Timer ( void ) {
}

void Timer::Start ( void ) {
	clock_gettime(CLOCK_REALTIME, &_timeStart);
}

void Timer::Stop ( void ) {
	clock_gettime(CLOCK_REALTIME, &_timeStop);
}

float Timer::ElapsedTime( void ) {
	return (_timeStop.tv_sec-_timeStart.tv_sec)*1e3 + (_timeStop.tv_nsec-_timeStart.tv_nsec)/1e6;
}
