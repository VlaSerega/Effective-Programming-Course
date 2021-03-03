#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#define Xa 0.0f
#define Xb 4.0f
#define Ya 0.0f
#define Yb 4.0f

#define Sx  1
#define Sy  1

#define F0  1.0f
#define T0  1.5f

#define Y 4.0f

void wave(int Nx, int Ny, int Nt) {
    FILE *fp;
    const double t = Nx <= 1000 && Ny <= 1000 ? 0.01f : 0.001f;
    const double t2 = t * t;
    double h2x2 = 2 * (Xb - Xa) / (Nx - 1) * (Xb - Xa) / (Nx - 1);
    double h2y2 = 2 * (Yb - Ya) / (Ny - 1) * (Yb - Ya) / (Nx - 1);
    double f;

    double *U_prev = (double *) malloc(sizeof(double) * Nx * Ny);
    double *U_cur = (double *) malloc(sizeof(double) * Nx * Ny);

    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            U_cur[i * Nx + j] = 0.0f;
            U_prev[i * Nx + j] = 0.0f;
        }
    }

    for (int T = 0; T < Nt; ++T) {
        f = exp(-(2 * M_PI * F0 * (T * t - T0)) * (2 * M_PI * F0 * (T * t - T0)) / (Y * Y)) *
            sin(2 * M_PI * F0 * (T * t - T0));
        for (int i = 1; i < Ny - 1; ++i) {
            for (int j = 1; j < Nx - 1; ++j) {
                double tmpf = i == Sy && j == Sx ? f : 0;
                double P_left = j - 1 < Nx / 2 ? 0.01f : 0.04f;
                double P_cur = j < Nx / 2 ? 0.01f : 0.04f;
                U_prev[i * Nx + j] = 2 * U_cur[i * Nx + j] - U_prev[i * Nx + j] +
                                     t2 * (tmpf + ((U_cur[i * Nx + j + 1] - U_cur[i * Nx + j]) * (P_cur + P_cur) +
                                                   (U_cur[i * Nx + j - 1] - U_cur[i * Nx + j]) * (P_left + P_left)) /
                                                  h2x2 +
                                           ((U_cur[(i + 1) * Nx + j] - U_cur[i * Nx + j]) * (P_left + P_cur) +
                                            (U_cur[(i - 1) * Nx + j] - U_cur[i * Nx + j]) * (P_left + P_cur)) / h2y2);
            }
        }
        double *tmp = U_cur;
        U_cur = U_prev;
        U_prev = tmp;
    }

    fp = fopen("new.dat", "wb");
    fwrite(U_cur, sizeof(double), Nx * Ny, fp);
    fclose(fp);
    free(U_cur);
    free(U_prev);
}

int main(int argc, char **argv) {
    int Nx;
    int Ny;
    int Nt;
    char *end;
    if (argc < 4 || argc > 4) {
        fprintf(stderr, "Wrong count of arguments\n");
        return 0;
    }

    Nx = (int) strtol(argv[1], &end, 10);
    Ny = (int) strtol(argv[2], &end, 10);
    Nt = (int) strtol(argv[3], &end, 10);

    struct timespec start, endt;
    clock_gettime(CLOCK_REALTIME_COARSE, &start);

    wave(Nx, Ny, Nt);

    clock_gettime(CLOCK_REALTIME_COARSE, &endt);

    printf("Sec: %f\n", (double) endt.tv_sec - start.tv_sec + ((double) endt.tv_nsec - start.tv_nsec) / 1000000000);

    return 0;
}
