#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "structs.h"

Tensor4 GetData4(const char *);
Tensor3 GetData3(const char *);
Matrix GetData2(const char *);
Tensor1 GetData1(const char *);
char *GetPath(int, const char *);

#endif