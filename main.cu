#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


// Оценим время выполнения задачи для разного числа городов N
// Умножение Xt * Q * X будет стоить N**4 флопов, т.к. матрица Q имеет размер N**2 х N**2
// Поле перебора всех значений X имеет размер 2**(N**2)
// Tesla A100 может 300 TFlops в FP16 tensor core, но данное решение не использует тензорные возможности железа
// а опирается на общего назначения FP32 ядра, это 20 TFlops.
// Архитектура Tesla A100 включает 108 Streaming Multiprocessor х 64 FP32 CUDA kernels
// 20TFlops = 2**34 flops

// Для N==4 объем задачи 2**16 * 4**4 = 2**24 флопов. Время выполнения 2**(24-34), одна миллисекунда
// Для N==5 объем задачи 2**25 * 5**4 = 2**35 флопов. Время выполнения 2**(35-34), одна секунда
// Для N==6 объем задачи 2**36 * 6**4 = 2**46 флопов. Время выполнения 2**(46-34), 4000 секунд, один час
// Для N==7 объем задачи 2**49 * 7**4 = 2**60 флопов. Время выполнения 2**(60-34), 60 млн секунд, два года
// Для N==8 объем задачи 2**64 * 8**4 = 2**76 флопов. Время выполнения 2**(76-34), 4* трлн секунд, 120 тысяч лет

// Поэтому будем решать задачу пока для N == 6, далее, если оптимизации окажутся заметными замахнемся на N == 7
// Поле перебора вектора X имеет размерность 2**36, 36 бит. Разобъем его таким образом:
// Младшие 8 бит (29..36) выполняются параллельно в 256 потоков блока, во внутреннем цикле ядра
// Следующие 13 бит (16..28)
// Следующие 9 бит (7..15) составят GridSize задачи. 2**9 почти кратно 108, будет хорошая occupancy на уровне SM
// Старшие 6 бит (1..6) будем перебирать в цикле хостовой части программы, чтобы иметь возможность выводить в консоль текущий прогресс


// Ядро, которое делает следующее
//      Операции выполняются один раз на старте ядра
// 1. Получает координаты городов, строит матрицу Q, это быстрее и проще сделать в ядре, чем передавать готовую
// 2. Получает фиксированные fixed_size старших бит вектора X
// 3. Считает Xt * Q * X для фиксированной части X
//      Выполняет цикл, перебирает по все значения переменной части вектора X
// 1. Берет младшие 8 бит переменной части (blockDim.x = 256)
// 2. В 256 потоков досчитывает Xt * Q * X для переменной части X
// 3. Выбирает минимум
// 4. Обновляет ранее найденный минимум, запоминает лучший X
//     По завершении цикла возвращает лучший минимум энергии и его X

/*
Умножение Xt * Q * X будем делать поблочно, здесь Xa - фиксированная часть X, Xb -переменная
X^T Q X = \begin{pmatrix}
Xa^T & Xb^T
\end{pmatrix}
\begin{pmatrix}
Q{aa} & Q{ab} \\
0 & Q{bb}
\end{pmatrix}
\begin{pmatrix}
Xa \\
Xb
\end{pmatrix} \]

\[ X^T Q X = Xa^T Q{aa} Xa + Xa^T Q{ab} Xb + Xb^T Q{bb} Xb \]

 \[ X^T Q X = C1 + C2 Xb + Xb^T Q{bb} Xb \]
*/

void evaluate_distances(
        int N,                  // число городов
        float* city_XYs,        // массив пар (X, Y) координат городов
        float* distances,       // NxN матрица дистанций между городами, является результатом работы функции
        float& max_dist         // максимальная дистанция, тоже результат
) {
    max_dist = 0;
    for (int i = 0; i != N; ++i) {
        float Xi = city_XYs[2 * i];
        float Yi = city_XYs[2 * i + 1];

        for (int j = 0; j != N; ++j) {
            float Xj = city_XYs[2 * j];
            float Yj = city_XYs[2 * j + 1];
            float dx = Xi - Xj;
            float dy = Yi - Yj;
            float dist = sqrtf(dx*dx + dy*dy);
            distances[threadIdx.x] = dist;
            if (dist > max_dist)
                max_dist = dist;
        }
    }
};

void evaluate_QUBO_matrix(
        int N,                   // число городов
        float* distances,        // NxN матрица дистанций между городами
        float* Q,                // N**2 x N**2 матрица QUBO, является результатом работы функции
        float& max_dist          // максимаьлная значение в матрице distances
) {
    for (int i = 0; i != N; ++i){
        for (int j = 0; j != N; ++j){
            Q[i*N + j] = 0;
            for (int p = 0; p != N-1; ++p){
                int idx1 = i*N + p;
                int idx2 = j*N + p + 1;
                int idx3 = idx1 * N * N + idx2;
                float dist = distances[i*N + j];
                Q[idx3] = dist;
            }
        }
    }
};

__global__ void bruteforce_semifixed_X(
        int N,                  // число городов
        float* city_XYs,        // массив пар (X, Y) координат городов
        int* fixed_bits,        // массив с фиксированной частью X ...
        int fixed_bits_cnt,     // ... и его длина
        int grid_bits_cnt,      // длина второй половины фиксированной части X
        int cycled_bits_cnt)    // число бит перебираемых в цикле
{
    if ((N < 4) || (N > 7))
        return;

    float distances[7*7];
    float Q[49 * 49]; // Матрица QUBO. N**2 x N**2
    float max_dist = 0;
    float C1 = 0; // Xat * Qaa * Xa
// Xa состоит из двух частей - fixed_bits, и часть определяемая по номеру блока
// Надо восстановить вторую часть
    int Xa[49];
    int Xa_len = fixed_bits_cnt + grid_bits_cnt;
    int Xb_len = N * N - Xa_len;
    for(int i = 0; i != fixed_bits_cnt; ++i){
        Xa[i] = fixed_bits[i];
    }
    int num = blockIdx.x;
    for(int i = 0; i != fixed_bits_cnt; ++i){
        Xa[Xa_len - i - 1] = num & 1;
        num >>= 1;
    }
// Найдем C1 как Xat * Qaa * Xa
    float temp[49];
    // Умножаем матрицу Q на вектор Xa
    for (int i = 0; i < Xa_len; i++) {
        temp[i] = 0.0;
        for (int j = 0; j < Xa_len; j++) {
            temp[i] += Q[ i*49+j ] * Xa[j];
        }
    }
    for (int i = 0; i < Xa_len; i++) {
        C1 += Xa[i] * temp[i];
    }

// Найдем C2 как Xat * Qab
    float C2[49]; // Xat * Qab
    for (int i = 0; i < Xb_len; i++) {
        C2[i] = 0;
        for (int j = 0; j < Xa_len; j++) {
            C2[i] += Xa[j] * Q[49 * j + i + Xa_len];
        }
    }

// Цикл по левой части вектора Xb. 2**cycled_bits_cnt итераций
    unsigned int cycles_cnt = 1;
    for (int i = 0; i < cycled_bits_cnt-1; ++i){
        cycles_cnt <<= 1;
    };
    for (int i = 0; i < cycles_cnt; ++i){
// Так, тут проблемка. Левая часть вектора Xb должна быть общая для всех потоков блока
// А правую - каждый поток строит сам, по своему threadIdx
// Тут нужен еще один уровень блочного умножения



    }


}

int main() {
    int N = 100000;
    int BLOCK_SIZE = 256;
    int *a, *b, *c; // Хост-память
    int *d_a, *d_b, *d_c; // Девайс-память
    int size = N * sizeof(int);
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Выделяем память на хосте
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    std::cout << "Started random" << std::endl;

    // Инициализируем массивы случайными числами
    for(int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    std::cout << "Started mem copy host -> device, size = " << size << std::endl;

    // Выделяем память на девайсе (GPU)
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Копируем данные из хост-памяти в девайс-память
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Создаем события для замера времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::cout << "Started vectorAdd kernel" << std::endl;

    // Выполняем сложение массивов на GPU
    int gridSize = (int)ceil((float)N / BLOCK_SIZE);
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_c, N);

    // Останавливаем таймер
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Started mem copy device -> host, size = " << size << std::endl;
    // Копируем результат обратно в хост-память
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Total run time GPU: " << milliseconds << " milliseconds\n";

    // Освобождаем ресурсы
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}