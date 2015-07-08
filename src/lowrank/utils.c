#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include "utils.h"

float _Complex* generateChessboard(int width, int height, int boardBlockWidth, int boardBlockHeight) {
    float _Complex* result = (float _Complex*)calloc(width * height, sizeof(float _Complex));
    for(int i =0; i < width; i++){
        for(int j=0; j < height; j++){
            if(i % (2 * boardBlockHeight) < boardBlockHeight && j % (2 * boardBlockWidth) < boardBlockWidth){
                result[j * width + i] = 1+I;
            }
        }
    }
    return result;
}

float _Complex* generateRandomMatrix (int width, int height) {
    float _Complex* result = (float _Complex*)calloc(width * height, sizeof(float _Complex));
    for(int i = 0; i < width * height; i++) {
        result[i] = randComplex(100);
    }
    return result;
}


char* diffMatrix(float _Complex *m1, float _Complex *m2, int width, int height) {
    for(int i = 0; i < width * height; i++){
        if(!complexEqual(m1[i],m2[i])) {
            char* info = (char*)malloc(sizeof(char) * 40);
            sprintf(info, "matrix diff at %d, %d", i % width, i / width);
            return info;
        }
    }
    return NULL;
}

void printMatrix(float _Complex *m, int width, int height) {
    for(int j = 0; j < height; j++){
        for(int i =0; i < width; i++){
            printf("%.2f+%.2fi ", __real__(m[j * width + i]), __imag__(m[j * width + i]));
        }
        printf("\n");
    }
}

void printMatrixPrecise(float _Complex *m, int width, int height) {
    for(int j = 0; j < height; j++){
        for(int i =0; i < width; i++){
            printf("%f+%fi ", __real__(m[j * width + i]), __imag__(m[j * width + i]));
        }
        printf("\n");
    }
}

float _Complex randComplex(int max) {
#ifdef FAKERANDCOMPLEX
    return FAKERANDCOMPLEX;
#else
    int sign1 = rand()/RAND_MAX > 0.5 ? 1 : -1;
    int sign2 = rand()/RAND_MAX > 0.5 ? 1 : -1;
    return sign1 * (float)rand()/(float)(RAND_MAX/max) + sign2 * (float)rand()/(float)(RAND_MAX/max) * I ;
#endif
}


