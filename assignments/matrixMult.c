#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 1000

int matrixA[MATRIX_SIZE][MATRIX_SIZE];
int matrixB[MATRIX_SIZE][MATRIX_SIZE];
int result[MATRIX_SIZE][MATRIX_SIZE];

void generateMatrix(int matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

void multiply(int startRow, int endRow) {
    for (int i = startRow; i <= endRow; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            result[i][j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, numProcesses;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    if (rank == 0) {
        generateMatrix(matrixA);
        generateMatrix(matrixB);
    }

    // Broadcast matrixB to all processes
    MPI_Bcast(matrixB, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide work among processes
    int rowsPerProcess = MATRIX_SIZE / numProcesses;
    int startRow = rank * rowsPerProcess;
    int endRow = startRow + rowsPerProcess - 1;
    if (rank == numProcesses - 1) {
        endRow += MATRIX_SIZE % numProcesses;
    }

    // Scatter matrixA rows among processes
    MPI_Scatter(
        matrixA, rowsPerProcess * MATRIX_SIZE, MPI_INT,
        &matrixA[startRow][0], rowsPerProcess * MATRIX_SIZE, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // Perform local matrix multiplication
    double startTime = MPI_Wtime();
    multiply(startRow, endRow);
    double endTime = MPI_Wtime();

    // Gather result from all processes
    MPI_Gather(
        &result[startRow][0], rowsPerProcess * MATRIX_SIZE, MPI_INT,
        result, rowsPerProcess * MATRIX_SIZE, MPI_INT,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
		printf("Execution Time: %.6f seconds\n", endTime - startTime);
    }

    MPI_Finalize();

    return 0;
}
