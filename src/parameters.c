#include "../include/parameters.h"
#include "../include/structs.h"
#include <stdio.h>
#include <stdlib.h>

Tensor4 GetData4(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("File not found: %s\n", path); exit(1); }

    int B, H, X, Y;
    fread(&B, sizeof(int), 1, f);
    fread(&H, sizeof(int), 1, f);
    fread(&X, sizeof(int), 1, f);
    fread(&Y, sizeof(int), 1, f);

    Tensor4 out = alloc_tensor4(B, H, X, Y);
    fread(out.data, sizeof(float), B*H*X*Y, f);

    fclose(f);
    return out;
}

Tensor3 GetData3(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("File not found: %s\n", path); exit(1); }

    int B, X, D;
    fread(&B, sizeof(int), 1, f);
    fread(&X, sizeof(int), 1, f);
    fread(&D, sizeof(int), 1, f);

    Tensor3 out = alloc_tensor3(B, X, D);
    fread(out.data, sizeof(float), B*X*D, f);

    fclose(f);
    return out;
}

Tensor1 GetData1(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("File not found: %s\n", path); exit(1); }

    int D;
    fread(&D, sizeof(int), 1, f);

    Tensor1 out = alloc_tensor1(D);
    fread(out.data, sizeof(float), D, f);

    fclose(f);
    return out;
}