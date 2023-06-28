#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 1000

void generateMatrix(int* matrix) {
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrix[i] = rand() % 100;
    }
}

void parallelMatrixSum(int* matrixA, int* matrixB, int* result, int numElements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elementsPerProcess = numElements / size;
    int extraElements = numElements % size;

    int* localA = (int*)malloc(sizeof(int) * (elementsPerProcess + extraElements));
    int* localB = (int*)malloc(sizeof(int) * (elementsPerProcess + extraElements));
    int* localResult = (int*)malloc(sizeof(int) * (elementsPerProcess + extraElements));

    MPI_Scatter(matrixA, elementsPerProcess, MPI_INT, localA, elementsPerProcess, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrixB, elementsPerProcess, MPI_INT, localB, elementsPerProcess, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < elementsPerProcess; i++) {
        localResult[i] = localA[i] + localB[i];
    }

    if (rank == 0 && extraElements > 0) {
        for (int i = 0; i < extraElements; i++) {
            localResult[elementsPerProcess + i] = matrixA[numElements - extraElements + i] +
                                                  matrixB[numElements - extraElements + i];
        }
    }

    MPI_Gather(localResult, elementsPerProcess, MPI_INT, result, elementsPerProcess, MPI_INT, 0, MPI_COMM_WORLD);

    free(localA);
    free(localB);
    free(localResult);
}

int main(int argc, char** argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* matrixA = NULL;
    int* matrixB = NULL;
    int* result = NULL;

    if (rank == 0) {
        matrixA = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
        matrixB = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
        result = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);

        generateMatrix(matrixA);
        generateMatrix(matrixB);
    }

    double startTime = MPI_Wtime();

    parallelMatrixSum(matrixA, matrixB, result, MATRIX_SIZE * MATRIX_SIZE);

    double endTime = MPI_Wtime();

    if (rank == 0) {
        printf("Execution time: %f seconds\n", endTime - startTime);

        free(matrixA);
        free(matrixB);
        free(result);
    }

    MPI_Finalize();

    return 0;
}
