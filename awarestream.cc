
//
// Very slightly smarter stream class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// What is it: a vector with push and pop operations added on top
// and some other stuff

#include "awarestream.h"

svmvolatile svm_mutex awarestream::fifolock;
svmvolatile fifolist awarestream::strfifo;


