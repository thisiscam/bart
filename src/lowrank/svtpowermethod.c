#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

#include "svtpowermethod.h"
#include "utils.h"


#define MAX_RAND_COMPLEX 10
#define ACCURACY 1e-13
#define complexSquare(c) ((__real__ c)*(__real__ c)+(__imag__ c)*(__imag__ c))

extern void normalizeVector (float _Complex *vector, int length, float _Complex denom);
extern float forbeniousNormalizeVector (float _Complex *vector, int length);


void blockwiseSVTPowerMethodSoft(int blockWidth, int blockHeight, int numBlocks, 
	float threshold, float _Complex *resultMatrix, float _Complex *originalMatrix) {
	int maxIterations = blockWidth * MAX(6, blockWidth);
	//  int count = 0 ;
	#pragma omp parallel
	{
		//init a random vector
		float _Complex *singularVectorBase = (float _Complex*)malloc(sizeof(float _Complex) * blockWidth * blockHeight);
		srand(time(NULL));
		for(int i=0; i < blockWidth * blockHeight; i++) {
			singularVectorBase[i] = randComplex(threshold * 2);
		}
		float _Complex *singularVector = singularVectorBase;
		float _Complex *holderVector = (float _Complex*)malloc(sizeof(float _Complex) * MAX(blockWidth, blockHeight));

		#pragma omp for
		for(int blockStartXY=0; blockStartXY < numBlocks * blockHeight * blockWidth; blockStartXY += blockHeight * blockWidth) {
			float matrixForbeniusNormSquared = 0;
			for(int j=0; j < blockHeight * blockWidth; j+=blockWidth) {
				for(int i=0; i < blockWidth; i++) {
					float _Complex entry = originalMatrix[blockStartXY + j + i];
					resultMatrix[blockStartXY + j + i] = entry;
					matrixForbeniusNormSquared += complexSquare (entry);
				}
			}

			singularVector = singularVectorBase;
			//Now we have blockXStart and blockYStart to be the start of each block
			do {
				if (matrixForbeniusNormSquared < threshold * threshold) {
					for(int j=0; j < blockHeight * blockWidth; j+=blockWidth) {
						for(int i=0; i < blockWidth; i++) {
							resultMatrix[blockStartXY + j+ i] = 
							originalMatrix[blockStartXY + j + i] -
							resultMatrix[blockStartXY + j + i];
						}
					}
					break;
				}
				//do maxIter of power iteration
				float oldS1 = 0;
				for(int iter=0; iter < maxIterations/2; iter++) {
					//refill singularVector with blockMatrixH * blockMatrix * singularVector
					float _Complex rowSum = 0;
					for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
						rowSum += resultMatrix[blockStartXY + multColNumber]
						 * singularVector[multColNumber];
					}
					for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
						float _Complex product = conj(resultMatrix[blockStartXY + multColNumber])
						 * rowSum;
						holderVector[multColNumber] = product;
					}
					for(int multRowNumber=blockWidth; multRowNumber < blockHeight * blockWidth; multRowNumber += blockWidth) {
						float _Complex rowSum = 0;
						for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
							rowSum += resultMatrix[blockStartXY + multRowNumber + multColNumber]
							 * singularVector[multColNumber];
						}
						for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
							float _Complex product = conj(resultMatrix[blockStartXY + multRowNumber + multColNumber])
							 * rowSum;
							holderVector[multColNumber] += product;
						}
					}
					rowSum = 0;
					for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
						rowSum += resultMatrix[blockStartXY + multColNumber]
						 * holderVector[multColNumber];
					}
					for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
						float _Complex product = conj(resultMatrix[blockStartXY + multColNumber])
						 * rowSum;
						singularVector[multColNumber] = product;
					}
					for(int multRowNumber=blockWidth; multRowNumber < blockHeight * blockWidth; multRowNumber += blockWidth) {
						float _Complex rowSum = 0;
						for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
							rowSum += resultMatrix[blockStartXY + multRowNumber + multColNumber]
							 * holderVector[multColNumber];
						}
						for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
							float _Complex product = conj(resultMatrix[blockStartXY + multRowNumber + multColNumber])
							 * rowSum;
							singularVector[multColNumber] += product;
						}
					}
					float norm = forbeniousNormalizeVector (singularVector, blockWidth);
					if (fabs(norm - oldS1) < ACCURACY) {
						break;
					} else {
						oldS1 = norm;
					}
				}

				//compute one corresponding column u of U(M=U*Sigma*VT) from singularVector and store in holderVector
				float uLength = 0;
				for(int multRowNumber=0; multRowNumber < blockHeight; multRowNumber += 1) {
					float _Complex rowSum = 0;
					for(int multColNumber=0; multColNumber < blockWidth; multColNumber++) {
						rowSum += resultMatrix[blockStartXY + multRowNumber * blockWidth + multColNumber]
						 * singularVector[multColNumber];
					}
					uLength += complexSquare (rowSum);
					holderVector[multRowNumber] = rowSum;
				}
				float currentSingularValue = sqrt(uLength);

				if (currentSingularValue < threshold) {
					// if (currentSingularValue == 0) {
					// 	srand(time(NULL));
					// 	for(int i=0; i < blockWidth; i++) {
					// 		singularVector[i] = randComplex(MAX_RAND_COMPLEX);
					// 	}
					// }
					//printf("%d %f\n",count, uLength);
					// count =0;
					for(int j=0; j < blockHeight * blockWidth; j+=blockWidth) {
						for(int i=0; i < blockWidth; i++) {
							resultMatrix[blockStartXY + j+ i] = 
							originalMatrix[blockStartXY + j + i] -
							resultMatrix[blockStartXY + j + i];
						}
					}
					break;
				}

				//subtract singluar valued matrix from A
				//that is, A = A - lambda * u * singluarVectorsH
				for(int j=0; j < blockHeight; j++) {
					for(int i=0; i < blockWidth; i++) {
						resultMatrix[blockStartXY + j * blockWidth + i] -= 
						holderVector[j] * conj(singularVector[i]) * (currentSingularValue - threshold) / currentSingularValue;
					}
				}
				matrixForbeniusNormSquared -= complexSquare (currentSingularValue);
				singularVector += blockWidth;
			} while(true); //TODO add max entries checking
		}
	}
}

inline void normalizeVector (float _Complex *vector, int length, float _Complex denom) {
	for(int i=0; i < length; i++) {
		vector[i] = vector[i] / denom;
	}
}

inline float forbeniousNormalizeVector (float _Complex *vector, int length) {
	float norm = 0;
	for (int i=0; i < length; i++) {
		norm += complexSquare(vector[i]);
	}
	norm = sqrt(norm);
	normalizeVector (vector, length, norm);
	return norm;
}
