#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>
#include <stdlib.h>
#include <cstdlib>

using namespace std;


int main(int argc, char** argv)
{	
	int rank, size;
	int count, number_thread;
	int i, j, t;
	int overage = 0; 
	int N = 800;

	double start, end;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	ifstream input1, input2, input3;
	ofstream result;

	// Исходные матрицы
	double* A = new double[N * N];
	double* B = new double[N * N];
	double* F = new double[N * N];

	// Промежуточная матрица
	double* D = new double[N * N];
	// Финальная матрица
	double* C = new double[N * N];

	count = N / size;
	if (rank == size - 1 && N % size != 0) {
		overage = N % size;
	}

	// нулевой процесс считывает данные и отправляет данные всем остальным процессам
	if (rank == 0) {
		cout << "Size = " << N << endl;		
		input1.open("input1.txt");
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				input1 >> A[i * N + j];
			}
		}
		input1.close();

		input2.open("input2.txt");
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				input2 >> B[i * N + j];
			}
		}
		input2.close();

		input3.open("input3.txt");
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				input3 >> F[i * N + j];
			}
		}
		input3.close();

		start = MPI_Wtime();
	}

	MPI_Bcast(&A[0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&B[0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&F[0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// перемножаем первые две матрицы
	for (i = rank * count; i < count * (rank + 1) + overage; i++) {
		for (j = 0; j < N; j++) {
			D[i * N + j] = 0;
			for (t = 0; t < N; t++) {
				D[i * N + j] += A[i * N + t] * B[t * N + j];
			}
		}
	}

	delete[] A;
	delete[] B;

	// перемножаем полученную и третью
	for (i = rank * count; i < count * (rank + 1) + overage; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = 0;
			for (t = 0; t < N; t++) {
				C[i * N + j] += D[i * N + t] * F[t * N + j];
			}
		}
	}

	delete[] D;
	delete[] F;

	//MPI_GATHERV разрешается принимать от каждого процесса переменное число элементов данных
	//массив, содержащий кол-во элементов, получаемых от каждого процесса
	int* rcounts = new int[size];
	//массив, где элемент определяет смещение относительно rcounts, в котором размещаются данные из процесса th_num
	int* displs = new int[size];

	for (number_thread = 0; number_thread < size - 1; number_thread++) {
		displs[number_thread] = number_thread * count * N;
		rcounts[number_thread] = count * N;
	}
	displs[size - 1] = (size - 1) * count * N;
	rcounts[size - 1] = (count + N % size) * N;
	
	double* resultC = new double[N * N];

	MPI_Gatherv(&C[rank * count * N], (count + overage) * N, MPI_DOUBLE, resultC, rcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete[] C;
	
	if (rank == 0) {
		end = MPI_Wtime();
		cout << "Total time = " << end - start << endl;
	}

	MPI_Finalize();

	delete[] resultC;

	return 0;
}
