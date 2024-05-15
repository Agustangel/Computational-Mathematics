#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


double f(double t, double x) { return x + t; }

double phi(double x) { return cos(3.14159265 * x); }

double psi(double t) { return exp(-t); }

int idx(int k, int m, int M) { return k * M + m; }

void fill_func(double (*f)(double, double), double *du_data, int M, int K,
               double tau, double h) {
  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M; ++m) {
      int i = idx(k, m, M);
      du_data[i] = f(k * tau, m * h);
    }
  }
}

void fill_init(double (*f)(double), double *arr, int N, double h) {
  for (int i = 0; i < N; ++i) {
    arr[i] = f(i * h);
  }
}

int get_rank_start(int rank, int size, int N) {
  int n_range = N / size;
  int mod_size = N % size;

  int rank_start = 0;
  if (rank < mod_size) {
    rank_start = rank * (n_range + 1);
  } else {
    rank_start = rank * n_range + mod_size;
  }

  return rank_start;
}

int get_rank_end(int rank, int size, int N) {
  int n_range = N / size;
  int mod_size = N % size;

  int rank_end = 0;
  int rank_start = get_rank_start(rank, size, N);
  if (rank < mod_size) {
    rank_end = rank_start + n_range;
  } else {
    rank_end = rank_start + n_range - 1;
  }

  return rank_end;
}

void data_to_file(const char *data_file_name, double *u, int M, int K) {
  FILE *fd = fopen(data_file_name, "a");
  if (fd) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        int i = idx(k, m, M);
        fprintf(fd, "%lf ", u[i]);
      }
      fprintf(fd, "\n");
    }
    fclose(fd);
  }
}

void file_print(int rank, int size, const char *data_file_name, double *u,
                int M, int K) {
  if (rank != 0) {
    int special_signal = 0;
    // блокирующий приём
    MPI_Recv(&special_signal, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
  data_to_file(data_file_name, u, M, K);
  if (rank != size - 1) {
    int special_signal = 69;
    // блокирующая отправка
    MPI_Send(&special_signal, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }
}

void krest(int rank, int size, double *u, double *du_data, int rank_M, int K,
           int M, double tau, double h) {
  MPI_Request request = MPI_REQUEST_NULL;
  MPI_Status status;
  for (int m = 1; m < rank_M; ++m)
    u[idx(1, m, rank_M)] = du_data[idx(0, m + rank * rank_M, M)] * tau +
                           (h - tau) / h * u[idx(0, m, rank_M)] +
                           tau / h * u[idx(0, m - 1, rank_M)];

  if (rank != size - 1)
    MPI_Isend(&u[idx(0, rank_M - 1, rank_M)], 1, MPI_DOUBLE, rank + 1, 1,
              MPI_COMM_WORLD, &request);

  if (rank != 0) {
    double u_prev = 0.0;
    MPI_Irecv(&u_prev, 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
    u[idx(1, 0, rank_M)] = du_data[idx(0, 0 + rank * rank_M, M)] * tau +
                           (h - tau) / h * u[idx(0, 0, rank_M)] +
                           tau / h * u_prev;
  }

  for (int k = 1; k < K - 1; ++k) {
    for (int m = 1; m < rank_M; ++m) {
      if (m < rank_M - 1)
        u[idx(k + 1, m, rank_M)] =
            du_data[idx(k, m + rank * rank_M, M)] * 2 * tau +
            u[idx(k - 1, m, rank_M)] +
            (u[idx(k, m - 1, rank_M)] - u[idx(k, m + 1, rank_M)]) * tau / h;
      else
        u[idx(k + 1, m, rank_M)] = du_data[idx(k, m + rank * rank_M, M)] * tau +
                                   (h - tau) / h * u[idx(k, m, rank_M)] +
                                   tau / h * u[idx(k, m - 1, rank_M)];
    }

    if (rank != size - 1) {
      MPI_Isend(&u[idx(k, rank_M - 1, rank_M)], 1, MPI_DOUBLE, rank + 1, 0,
                MPI_COMM_WORLD, &request);
    }

    if (rank != 0) {
      double u_prev = 0.0;
      MPI_Irecv(&u_prev, 1, MPI_DOUBLE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD,
                &request);
      MPI_Wait(&request, &status);
      u[idx(k + 1, 0, rank_M)] =
          du_data[idx(k, 0 + rank * rank_M, M)] * 2 * tau +
          u[idx(k - 1, 0, rank_M)] + (u_prev - u[idx(k, 1, rank_M)]) * tau / h;
    }
  }
}

int main(int argc, char *argv[]) {
  double T = 1;
  double X = 1;

  if (argc != 3) {
    fprintf(stderr, "[-] Usage mpiexec -n 4 %s K N\n", argv[0]);
    return -1;
  }
  int K = atoi(argv[1]);
  int M = atoi(argv[2]);

  double tau = T / K;
  double h = X / M;

  double *du_data = (double *)calloc(K * M, sizeof(double));
  fill_func(f, du_data, M, K, tau, h);

  double *phi_data = (double *)calloc(K, sizeof(double));
  double *psi_data = (double *)calloc(M, sizeof(double));
  fill_init(psi, psi_data, K, tau);
  fill_init(phi, phi_data, M, h);

  int size = 0;
  int rank = 0;

  MPI_Status status;

  int rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    perror("[-] Error starting MPI program. Programm was terminated.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  double start = MPI_Wtime();

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rank_M_start = get_rank_start(rank, size, M);
  int rank_M_end = get_rank_end(rank, size, M);
  int rank_M = abs(rank_M_end - rank_M_start + 1);

  double *u = (double *)calloc(K * rank_M, sizeof(double));
  for (int i = rank_M_start; i <= rank_M_end; ++i)
    u[idx(0, i - rank_M * rank, rank_M)] = phi_data[i];
  if (rank == 0) {
    for (int i = 0; i < K; ++i)
      u[idx(i, 0, rank_M)] = psi_data[i];
  }

  krest(rank, size, u, du_data, rank_M, K, M, tau, h);

  const char *data_file_name = "data.txt";
  file_print(rank, size, data_file_name, u, rank_M, K);

  double end = MPI_Wtime();
  double rank_time = (end - start) * 1000; // в мс

  if (rank == size - 1) {
    const char *time_file_name = "time.txt";
    FILE *time_out_file = fopen(time_file_name, "a");
    if (time_out_file)
      fprintf(time_out_file, "%lf ", rank_time);
    fclose(time_out_file);
  }

  free(u);
  MPI_Finalize();

  free(psi_data);
  free(phi_data);
  free(du_data);
  return 0;
}
