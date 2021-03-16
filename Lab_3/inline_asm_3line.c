#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define Xa 0.0f
#define Xb 4.0f
#define Ya 0.0f
#define Yb 4.0f

#define Sx  1
#define Sy  1

#define F0  1.0f
#define T0  1.5f

#define Y 4.0f

void __attribute__ ((noinline))
count_line(double *U_cur, double *U_prev, double *P, int Nx, int i, double h2x2t2, double h2y2t2, double ft) {
    register double *tmpPrev = U_prev + Nx * i + 1;
    register double *tmpCur = U_cur + Nx * i + 1;
    register double *tmpP = P + Nx * i + 1;
    register double Ucur;
    register double *endCur = tmpCur + Nx - 2;
    double save = U_prev[Sy * Nx + Sx];

    __asm__ __volatile__(".intel_syntax noprefix\n\t"
                         "mov eax, %3\n\t"
                         "sub rax, 6\n\t"
                         "shl rax, 3\n\t"

                         "mov rbx, %1\n\t"
                         "add rbx, rax\n\t"

                         "mov eax, %3\n\t"
                         "shl rax, 3\n\t"

                         "jmp .end_cycle\n\t"
                         ".cycle:\n\t"
                         "vmovupd ymm0, [%1 - 8]\n\t"      //U_left
                         "vmovupd ymm1, [%1]\n\t"           //U_cur
                         "vmovupd ymm2, [%1+8]\n\t"     //U_right

                         "vmovupd ymm3, [%1 + rax]\n\t"     //U_low
                         "mov rcx, %1\n\t"
                         "sub rcx, rax\n\t"
                         "vmovupd ymm4, [rcx]\n\t"     //U_top

                         "vmovupd ymm5, [%2]\n\t"     //P_cur
                         "mov rcx, %2\n\t"
                         "sub rcx, rax\n\t"
                         "vmovupd ymm6, [rcx]\n\t"     //P_top
                         "vmovupd ymm7, [%2 - 8]\n\t"     //P_left
                         "vmovupd ymm8, [rcx - 8]\n\t"     //P_top_left

                         "vsubpd ymm10, ymm2, ymm1\n\t"  //U_right - U_cur
                         "vaddpd ymm11, ymm6, ymm5\n\t" //P_top + P_cur
                         "vmulpd ymm10, ymm10, ymm11\n\t" //(U_right - U_cur) * (P_top + P_cur)

                         "vsubpd ymm11, ymm0, ymm1\n\t"  //U_left - U_cur
                         "vaddpd ymm12, ymm8, ymm7\n\t" //P_top_left + P_left
                         "vfmadd231pd ymm10, ymm11, ymm12\n\t" //(U_left - U_cur) * (P_top_left + P_cur) + (U_right - U_cur) * (P_top + P_cur)
                         "vmovddup xmm9, %7\n\t"
                         "vinsertf128 ymm9, ymm9, xmm9, 0x1\n\t"
                         "vmulpd ymm10, ymm10, ymm9\n\t" //Previous result * h2x2t2   (1)

                         "vsubpd ymm11, ymm3, ymm1\n\t"  //U_low - U_cur
                         "vaddpd ymm12, ymm7, ymm5\n\t" //P_left + P_cur
                         "vmulpd ymm11, ymm11, ymm12\n\t" //(U_low - U_cur) * (P_left + P_cur)

                         "vsubpd ymm12, ymm4, ymm1\n\t"  //U_top - U_cur
                         "vaddpd ymm13, ymm8, ymm6\n\t" //P_top_left + P_top
                         "vfmadd231pd ymm11, ymm12, ymm13\n\t" //(U_top - U_cur) * (P_top_left + P_top) + (U_low - U_cur) * (P_left + P_cur)
                         "vmovddup xmm9, %8\n\t"
                         "vinsertf128 ymm9, ymm9, xmm9, 0x1\n\t"
                         "vfmadd231pd ymm10, ymm11, ymm9\n\t"  //(1) + previous result * h2y2t2

                         "vmovupd ymm9, [%0]\n\t"      //U_prev

                         "vaddpd ymm11, ymm1, ymm1\n\t"  //2*U_cur
                         "vsubpd ymm11, ymm11, ymm9\n\t"
                         "vaddpd ymm10, ymm11, ymm10\n\t"
                         "vmovupd [%0], ymm10\n\r"

                         "add %0, 32\n\t"
                         "add %1, 32\n\t"
                         "add %2, 32\n\t"

                         ".end_cycle:\n\t"
                         "cmp %1, rbx\n\t"
                         "jle .cycle\n\t"

                         ".att_syntax \n\t"
    : "=r"(tmpPrev), "=r"(tmpCur), "=r"(tmpP)
    :"r"(Nx), "0"(tmpPrev), "1"(tmpCur), "2"(tmpP), "x"(h2x2t2), "x"(h2y2t2)
    : "rax", "rbx", "rcx", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13"
    );

    while (tmpCur != endCur) {
        Ucur = tmpCur[0];
        *tmpPrev = 2 * Ucur - *tmpPrev +
                   ((tmpCur[1] - Ucur) * (tmpP[-Nx] + tmpP[0]) +
                    (tmpCur[-1] - Ucur) * (tmpP[-Nx - 1] + tmpP[-1])) * h2x2t2 +
                   ((tmpCur[Nx] - Ucur) * (tmpP[-1] + tmpP[0]) +
                    (tmpCur[-Nx] - Ucur) * (tmpP[-Nx - 1] + tmpP[-Nx])) * h2y2t2;
        tmpPrev++;
        tmpCur++;
        tmpP++;
    }

    if (i == Sy) {

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
    }
}

void wave(int Nx, int Ny, int Nt) {
    FILE *fp;
    const double t = Nx <= 1000 && Ny <= 1000 ? 0.01f : 0.001f;
    register const double h2x2t2 = t * t / (2 * (Xb - Xa) / (double) (Nx - 1) * (Xb - Xa) / (Nx - 1));
    register const double h2y2t2 = t * t / (2 * (Yb - Ya) / (double) (Ny - 1) * (Yb - Ya) / (Nx - 1));

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

    for (int T = 0; T < Nt; T += 3) {
        double ft1 = exp(-(2 * M_PI * F0 * (T * t - T0)) * (2 * M_PI * F0 * (T * t - T0)) / (Y * Y)) *
                     sin(2 * M_PI * F0 * (T * t - T0)) * t * t;
        double ft2 = exp(-(2 * M_PI * F0 * ((T + 1) * t - T0)) * (2 * M_PI * F0 * ((T + 1) * t - T0)) / (Y * Y)) *
                     sin(2 * M_PI * F0 * ((T + 1) * t - T0)) * t * t;
        double ft3 = exp(-(2 * M_PI * F0 * ((T + 2) * t - T0)) * (2 * M_PI * F0 * ((T + 2) * t - T0)) / (Y * Y)) *
                     sin(2 * M_PI * F0 * ((T + 2) * t - T0)) * t * t;

        count_line(U_cur, U_prev, P, Nx, 1, h2x2t2, h2y2t2, ft1);
        count_line(U_cur, U_prev, P, Nx, 2, h2x2t2, h2y2t2, ft1);
        count_line(U_prev, U_cur, P, Nx, 1, h2x2t2, h2y2t2, ft2);

        for (int i = 3; i < Nx - 1; ++i) {
            count_line(U_cur, U_prev, P, Nx, i, h2x2t2, h2y2t2, ft1);
            count_line(U_prev, U_cur, P, Nx, i - 1, h2x2t2, h2y2t2, ft2);
            count_line(U_cur, U_prev, P, Nx, i - 2, h2x2t2, h2y2t2, ft3);
        }

        count_line(U_prev, U_cur, P, Nx, Ny - 2, h2x2t2, h2y2t2, ft2);
        count_line(U_cur, U_prev, P, Nx, Ny - 3, h2x2t2, h2y2t2, ft3);
        count_line(U_cur, U_prev, P, Nx, Ny - 2, h2x2t2, h2y2t2, ft3);
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
