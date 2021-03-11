#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

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
    register const double h2x2t2 = t * t / (2 * (Xb - Xa) / (double) (Nx - 1) * (Xb - Xa) / (Nx - 1));
    register const double h2y2t2 = t * t / (2 * (Yb - Ya) / (double) (Ny - 1) * (Yb - Ya) / (Nx - 1));
    long rowLength = ((Nx / 4) + (Nx % 4 > 0)) * 4;
    register double ft;
    register double *tmpPrev;
    register double *tmpCur;
    register double *tmpP;

    double *U_prev = (double *) _mm_malloc(sizeof(double) * rowLength * Ny, 32);
    double *U_cur = (double *) _mm_malloc(sizeof(double) * rowLength * 4 * Ny, 32);
    double *P = (double *) _mm_malloc(sizeof(double) * rowLength * Ny, 32);

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

        ft = exp(-(2 * M_PI * F0 * (T * t - T0)) * (2 * M_PI * F0 * (T * t - T0)) / (Y * Y)) *
             sin(2 * M_PI * F0 * (T * t - T0)) * t * t;

        for (int i = 1; i < Ny - 1; ++i) {
            tmpPrev = U_prev + rowLength * i + 1;
            tmpCur = U_cur + rowLength * i + 1;
            tmpP = P + rowLength * i + 1;
            register double *endCur = tmpCur + rowLength - 1;

            for (int j = 0; j < 3 && j < Nx; ++j) {
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

            __asm__ __volatile__(".intel_syntax noprefix\n\t"
                                 "mov rax, %1\n\t"
                                 "sub rax, %4\n\t"

                                 "mov rcx, %2\n\t"
                                 "sub rcx, %4\n\t"

                                 "vmovapd ymm0, [%1 - 32]\n\t"  //U выравненный левый
                                 "vmovapd ymm1, [%1]\n\t"       //U выравненный текущий
                                 "vmovapd ymm2, [%1 + 32]\n\t"  //U выравненный правый
                                 "vmovapd ymm5, [%2 - 32]\n\t"  //P выравненный левый
                                 "vmovapd ymm6, [%2]\n\t"       //P выравненный текущий
                                 "vmovapd ymm7, [rcx]\n\t"      //P выравненный верхний
                                 "vmovapd ymm8, [rcx-32]\n\t"   //P выравненный верхний левый

                                 "movd xmm13, %5\n\t"
                                 "movddup xmm13, xmm13\n\t"
                                 "vmovddup ymm13, ymm13\n\t"

                                 "movd xmm14, %6\n\t"
                                 "movddup xmm14, xmm14\n\t"
                                 "vmovddup ymm14, ymm14\n\t"

                                 "jmp .end_cycle\n\t"
                                 ".cycle:\n\t"

                                 "vextractf128 xmm9, ymm0, 0x1\n\t" //Берем верхние два элемента из левого U
                                 "vpermilpd xmm10, xmm1, 0x1\n\t"    //Меняем местами два нижних элемента из текущего U
                                 "vpermilpd xmm9, xmm9, 0x1\n\t"    //Меняем местами два верхних эелемента  из левого U
                                 "blendpd xmm9, xmm10, 0x2\n\t"      //Собираем нижние два элемента из U_left
                                 "vextractf128 xmm11, ymm1, 0x1\n\t" //Берем верхние два элемента из текущего U
                                 "vpermilpd xmm11, xmm11, 0x1\n\t"    //Меняем местами два верхних элемента из текущего U
                                 "blendpd xmm10, xmm11, 0x2\n\t"      //Собираем верхние два элемента из U_left
                                 "vinsertf128 ymm9, ymm9, xmm10, 0x1\n\t"//Собираем U_left

                                 "vextractf128 xmm12, ymm1, 0x1\n\t" //Берем верхние два элемента из текущего U
                                 "vpermilpd xmm11, xmm2, 0x1\n\t"   //Меняем местами два нижних элемента из правого U
                                 "vpermilpd xmm12, xmm12, 0x1\n\t"    //Свапнули верхние два элемента из текущего U
                                 "movupd xmm10, xmm12\n\t"           //Сохраняем
                                 "blendpd xmm12, xmm11, 0x2\n\t"     //Собираем верхние два элемента из U_right
                                 "vpermilpd xmm11, xmm1, 0x1\n\t"   //Меняем местами два нижних элемента из текущего U
                                 "blendpd xmm11, xmm10, 0x2\n\t"    //Собираем нижние два эелемента из U_right
                                 "vinsertf128 ymm10, ymm11, xmm12, 0x1\n\t"//СОбираем U_right


                                 "vextractf128 xmm11, ymm5, 0x1\n\t" //Берем верхние два элемента из левого P
                                 "vpermilpd xmm12, xmm6, 0x1\n\t"    //Меняем местами два нижних элемента из текущего P
                                 "vpermilpd xmm11, xmm11, 0x1\n\t"    //Меняем местами два верхних эелемента  из левого P
                                 "blendpd xmm11, xmm12, 0x2\n\t"      //Собираем нижние два элемента из P_left
                                 "vextractf128 xmm3, ymm6, 0x1\n\t" //Берем верхние два элемента из текущего P
                                 "vpermilpd xmm3, xmm3, 0x1\n\t"    //Меняем местами два верхних элемента из текущего P
                                 "blendpd xmm12, xmm3, 0x2\n\t"      //Собираем верхние два элемента из P_left
                                 "vinsertf128 ymm11, ymm11, xmm12, 0x1\n\t"//Собираем P_left

                                 "vextractf128 xmm12, ymm8, 0x1\n\t" //Берем верхние два элемента из левого верхнего P
                                 "vpermilpd xmm3, xmm7, 0x1\n\t"    //Меняем местами два нижних элемента из верхнего P
                                 "vpermilpd xmm12, xmm12, 0x1\n\t"    //Меняем местами два верхних эелемента  из левого верхнего P
                                 "blendpd xmm12, xmm3, 0x2\n\t"      //Собираем нижние два элемента из P_top_left
                                 "vextractf128 xmm4, ymm7, 0x1\n\t" //Берем верхние два элемента из верхнего P
                                 "vpermilpd xmm4, xmm4, 0x1\n\t"    //Меняем местами два верхних элемента из верхнего P
                                 "blendpd xmm3, xmm4, 0x2\n\t"      //Собираем верхние два элемента из P_top_left
                                 "vinsertf128 ymm12, ymm12, xmm3, 0x1\n\t"//Собираем P_top_left


                                 "vsubpd ymm10, ymm10, ymm1\n\t"  //U_right - U_cur
                                 "vaddpd ymm3, ymm7, ymm6\n\t" //P_top + P_cur
                                 "vmulpd ymm10, ymm10, ymm13\n\t" //(U_right - U_cur) * (P_top + P_cur)

                                 "vsubpd ymm9, ymm9, ymm1\n\t"  //U_left - U_cur
                                 "vaddpd ymm3, ymm12, ymm11\n\t" //P_top_left + P_left
                                 "vfmadd231pd ymm10, ymm9, ymm3\n\t" //(U_left - U_cur) * (P_top_left + P_left) + (U_right - U_cur) * (P_top + P_cur)
                                 "vmulpd ymm10, ymm10, ymm13\n\t" //Previous result * h2x2t2   (1)

                                 "vmovapd ymm3, [%1 + %4]\n\t"     //U_low
                                 "vmovapd ymm4, [rax]\n\t"     //U_top

                                 "vsubpd ymm3, ymm3, ymm1\n\t"  //U_low - U_cur
                                 "vaddpd ymm11, ymm11, ymm6\n\t" //P_left + P_cur
                                 "vmulpd ymm3, ymm3, ymm11\n\t" //(U_low - U_cur) * (P_left + P_cur)

                                 "vsubpd ymm4, ymm4, ymm1\n\t"  //U_top - U_cur
                                 "vaddpd ymm12, ymm12, ymm7\n\t" //P_top_left + P_top
                                 "vfmadd231pd ymm3, ymm12, ymm4\n\t" //(U_top - U_cur) * (P_top_left + P_top) + (U_low - U_cur) * (P_left + P_cur)
                                 "vfmadd231pd ymm10, ymm3, ymm14\n\t"  //(1) + previous result * h2y2t2

                                 "vmovapd ymm9, [%0]\n\t"      //U_prev

                                 "vaddpd ymm11, ymm1, ymm1\n\t"  //2*U_cur
                                 "vsubpd ymm11, ymm11, ymm9\n\t"
                                 "vaddpd ymm10, ymm11, ymm10\n\t"
                                 "vmovntpd [%0], ymm10\n\r"

                                 "add %0, 32\n\t"
                                 "add %1, 32\n\t"
                                 "add %2, 32\n\t"
                                 "add rax, 32\n\t"
                                 "add rcx, 32\n\t"

                                 "vmovapd ymm0, ymm1\n\t"
                                 "vmovapd ymm1, ymm2\n\t"
                                 "vmovapd ymm2, [%1+32]\n\t"

                                 "vmovapd ymm5, ymm6\n\t"
                                 "vmovapd ymm6, [%2 + 32]\n\t"

                                 "vmovapd ymm7, ymm8\n\t"
                                 "vmovapd ymm8, [rcx]\n\t"

                                 ".end_cycle:\n\t"
                                 "cmp %1, %3\n\t"
                                 "jne .cycle\n\t"

                                 ".att_syntax \n\t"
            ::"r"(tmpPrev), "r"(tmpCur), "r"(tmpP), "r"(endCur), "r"(rowLength * 8), "r"(h2x2t2), "r"(h2y2t2)
            : "rax", "rcx", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14"
            );
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
    _mm_free(U_cur);
    _mm_free(U_prev);
    _mm_free(P);
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
