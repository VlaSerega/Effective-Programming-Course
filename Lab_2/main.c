#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>

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
    register const double h2x2t2 = t * t / (2 * (Xb - Xa) / (Nx - 1) * (Xb - Xa) / (Nx - 1));
    register const double h2y2t2 = t * t / (2 * (Yb - Ya) / (Ny - 1) * (Yb - Ya) / (Nx - 1));
    register double ft;
    register double *tmpPrev;
    register double *tmpCur;
    register double *tmpP;

    double *U_prev = (double *) malloc(sizeof(double) * Nx * Ny);
    double *U_cur = (double *) malloc(sizeof(double) * Nx * Ny);
    double *P = (double *) malloc(sizeof(double) * Nx * Ny);

    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            U_cur[i * Nx + j] = 0.0f;
            U_prev[i * Nx + j] = 0.0f;
            P[i * Nx + j] = j < Nx / 2 ? 0.01 : 0.04;
        }
    }

    for (int T = 0; T < Nt; ++T) {
        double save = U_prev[Sy * Nx + Sx];
        register double Ucur;

        tmpPrev = U_prev + Nx + 1;
        tmpCur = U_cur + Nx + 1;
        tmpP = P + Nx + 1;
        ft = exp(-(2 * M_PI * F0 * (T * t - T0)) * (2 * M_PI * F0 * (T * t - T0)) / (Y * Y)) *
             sin(2 * M_PI * F0 * (T * t - T0)) * t * t;

        for (int i = 1; i < Ny - 1; ++i) {
            register double Uleft = tmpCur[-1];

            Ucur = tmpCur[0];

            for (register int j = 1; j < Nx - 1; ++j) {
                register double Uright = tmpCur[1];
                *tmpPrev = 2 * Ucur - *tmpPrev +
                           ((Uright - Ucur) * (tmpP[-Nx] + tmpP[0]) +
                            (Uleft - Ucur) * (tmpP[-Nx - 1] + tmpP[-1])) * h2x2t2 +
                           ((tmpCur[Nx] - Ucur) * (tmpP[-1] + tmpP[0]) +
                            (tmpCur[-Nx] - Ucur) * (tmpP[-Nx - 1] + tmpP[-Nx])) * h2y2t2;
                tmpPrev++;
                tmpCur++;
                tmpP++;
                Uleft = Ucur;
                Ucur = Uright;
            }

            tmpPrev += 2;
            tmpCur += 2;
            tmpP += 2;
        }

        Ucur = U_cur[Sy * Nx + Sx];
        U_prev[Sy * Nx + Sx] = 2 * Ucur - save +
                               ((U_cur[Sy * Nx + Sx + 1] - Ucur) *
                                (P[(Sy - 1) * Nx + Sx] + P[Sy * Nx + Sx]) +
                                (U_cur[Sy * Nx + Sx - 1] - Ucur) *
                                (P[(Sy - 1) * Nx + Sx - 1] + P[Sy * Nx + Sx - 1])) *
                               h2x2t2 +
                               ((U_cur[(Sy + 1) * Nx + Sx] - Ucur) *
                                (P[Sy * Nx + Sx - 1] + P[Sy * Nx + Sx]) +
                                (U_cur[(Sy - 1) * Nx + Sx] - Ucur) *
                                (P[(Sy - 1) * Nx + Sx - 1] + P[(Sy - 1) * Nx + Sx])) *
                               h2y2t2 + ft;

        double *tmp = U_cur;
        U_cur = U_prev;
        U_prev = tmp;
    }

    fp = fopen("new.dat", "wb");
    fwrite(U_cur, sizeof(double), Nx * Ny, fp);
    fclose(fp);
    free(U_cur);
    free(U_prev);
    free(P);
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
