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
// 1. Берет младшие 8 бит переменной части (blockDim.x == 256)
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

Переменные в ядре будут называться по другому
 Qaa - Q11
 Qab - Q12
 Qbb - Q22
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
            distances[i*N + j] = dist;
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
    // Уф, тут еще дофига надо написать, пока так
};


void make_Q_matrix(int N,
                   int randseed,
                   float* Q) {
    auto city_XYs = new float[2 * N];

    std::srand(randseed);
    for (int i = 0; i < N; ++i) {
        city_XYs[2 * i] = static_cast<float>(std::rand()) / RAND_MAX; // Normalize to [0, 1]
        city_XYs[2 * i + 1] = static_cast<float>(std::rand()) / RAND_MAX; // Normalize to [0, 1]
    }
    float max_dist;
    auto dists = new float[N * N];
    evaluate_distances(N, city_XYs, dists,max_dist);
    // Инициализация матрицы Q
    evaluate_QUBO_matrix(N, dists, Q, max_dist);

}

__constant__ float constQ[49 * 49]; // Поскольку матрица Q одна и та же для всех ядер, положим ее в константную память

__global__ void findBestXKernel(
//       const int N,                     // Размер вектора X и матрицы Q. Не передаем, вычислим в коде
        const float* Q,              // Матрица QUBO. N**2 x N**2
        const uint64_t prefixC1,           // массив с фиксированной частью X ...
        const int C1,                      // ... и его длина
        const int C2,                      // длина второй половины фиксированной части X
        const int C3,
        uint64_t* bestX,             // здесь будем возвращать лучший найденный X
        float* bestE                // здесь будем возвращать лучшее найденное Е
        )                      // число бит перебираемых в цикле
{
    int C4 = 8;
    int N = C1 + C2 + C3 + C4;

    if ((N < 16) || (N > 49))
        return;

    float QProduct = 0; // Xat * Q11 * Xa
// Xa состоит из двух частей - C1 и С2,
// Надо восстановить вторую часть
    int C1C2 = C1 + C2;
    uint64_t fullPrefix = (prefixC1 << C2) | (blockIdx.x & ((1ULL << C2) - 1));

    __shared__ int prefixArray[49]; // Shared memory для вектора fullPrefix
    __shared__ float Q11Product; // Здесь будет храниться [fullPrefix]t * Q11 * [fullPrefix]
    __shared__ float Q12Vector[49]; // Здесь будет храниться [fullPrefix]t * Q12

    int tid = threadIdx.x;

    // Шаг 1. Развертываем fullPrefix в массив интов
    if (tid < C1 + C2) {
        prefixArray[tid] = (fullPrefix >> (C1 + C2 - 1 - tid)) & 1ULL;
    }
    __syncthreads(); // Синхронизация для обеспечения заполнения массива

    // Шаг 2. Вычисляем [fullPrefix]t * Q11 * [fullPrefix]
    if (tid == 0) { // Пусть только один поток выполнит эту задачу
        Q11Product = 0.0f;
        for (int i = 0; i < C1 + C2; ++i) {
            for (int j = i; j < C1 + C2; ++j) { // Q11 верхнетреугольная
                Q11Product += prefixArray[i] * Q[i * N + j] * prefixArray[j];
            }
        }
    }
    __syncthreads(); // Синхронизация для обеспечения вычисления произведения

    // Шаг 3. Вычисляем вектор [fullPrefix]t * Q12
    if (tid < N) { // Вектор Q12 может быть больше, чем C1+C2
        Q12Vector[tid] = 0.0f;
        for (int i = 0; i < C1 + C2; ++i) {
            Q12Vector[tid] += prefixArray[i] * Q[i * N + tid];
        }
    }
    __syncthreads(); // Синхронизация для обеспечения вычисления вектора

    // Теперь у нас есть Q11Product и Q12Vector, и мы можем выполнить перебор всех подвекторов длины C3.
    __shared__ int subVectorC3[49]; // Будет использоваться для хранения текущего подвектора C3

    // Организуем цикл по всем возможным значениям подвектора C3
    const uint64_t maxC3Value = (1ULL << C3) - 1; // Максимальное значение для подвектора C3
    for (uint64_t subVecValue = 0; subVecValue <= maxC3Value; ++subVecValue) {
        if (tid < C3) { // Каждый поток заполняет часть массива subVectorC3
            subVectorC3[tid] = (subVecValue >> (C3 - 1 - tid)) & 1;
        }
        __syncthreads(); // Синхронизация для обеспечения заполнения массива

        int suffixC4 = threadIdx.x; // threadIdx.x - это уже бинарное представление суффикса длиной C4=8
        unsigned int X_3_4 = 0;

        for (int i = 0; i < C3; ++i) {
            X_3_4 |= subVectorC3[i] << (C4 + i);
        }
        X_3_4 |= suffixC4; // Добавляем суффикс C4 к X_3_4

        // Вычисляем E для каждого потока
        float E = 0.0f;
        int index = C1C2 + C3; // Индекс начала матрицы Q22
        for (int i = 0; i < C3 + C4; ++i) {
            if ((X_3_4 >> i) & 1) { // Если i-й бит X_3_4 установлен
                for (int j = i; j < C3 + C4; ++j) {
                    if ((X_3_4 >> j) & 1) {
                        E += Q[(index + i) * N + index + j]; // Вычисляем элемент E
                    }
                }
            }
        }

        // Шаг 4: Суммируем E с предвычисленными значениями и сохраняем в локальном массиве
        __shared__ float localE[256]; // Предполагаем, что размер блока не больше 256
        E += Q11Product; // Добавляем предвычисленное значение для fullPrefix
        for (int i = 0; i < N - C1C2; ++i) { // Добавляем вклад от Q12Vector и subVectorC3
            E += Q12Vector[C1C2 + i] * subVectorC3[i];
        }
        localE[threadIdx.x] = E;
        __syncthreads(); // Синхронизируем потоки перед следующими шагами

        // Тут найдем минимум в localE

        __syncthreads(); // Синхронизация перед следующей итерацией цикла
    }


}


// Функция для вывода прогресса
void printProgress(uint64_t current, uint64_t total) {
    float progress = (float)current / total * 100.0f;
    std::cout << "Progress: " << progress << "%" << std::endl;
}


int main(int argc, char *argv[]) {
    int randseed = 42;
    int Cnt = 4;
    if (argc > 1) randseed = atoi(argv[1]);
    if (argc > 2) Cnt = atoi(argv[2]);

    const int N = Cnt * Cnt;
    const int C1 = 6;
    const int C2 = 9;
    const int C3 = 13;
    const int C4 = 8;

    // Выделяем память для матрицы Q на хосте
    auto Q = new float[N * N];
    make_Q_matrix(Cnt, randseed, Q);

    // Выделяем память для матрицы Q на девайсе
    cudaMemcpyToSymbol(constQ, Q, N*N);

    // Подготовка к запуску кернела
    uint64_t* d_bestX;
    float* d_bestE;
    cudaMalloc(&d_bestX, sizeof(uint64_t));
    cudaMalloc(&d_bestE, sizeof(float));

    // Инициализация лучших значений
    uint64_t bestX = 0;
    float bestE = FLT_MAX;
    cudaMemcpy(d_bestE, &bestE, sizeof(float), cudaMemcpyHostToDevice);

    // Цикл перебора старших бит
    const int C1_cycles_count = 1 << C1;
    const int gridSize = 1 << C2;
    const int blockSize = 1 << C4; //256

    for (uint64_t i = 0; i < C1_cycles_count; ++i) {
        // Печать прогресса
        printProgress(i, C1_cycles_count);

        // Запуск кернела
        findBestXKernel<<< gridSize, blockSize>>>(constQ, i, C1, C2, C3, d_bestX, d_bestE);

        // Синхронизация
        cudaDeviceSynchronize();

        // Обновление лучшего результата
        float currentE;
        cudaMemcpy(&currentE, d_bestE, sizeof(float), cudaMemcpyDeviceToHost);
        if (currentE < bestE) {
            bestE = currentE;
            cudaMemcpy(&bestX, d_bestX, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }
    }

    // Вывод результата
    std::cout << "Best E: " << bestE << "\nBest X: " << bestX << std::endl;

    // Освобождение ресурсов
    cudaFree(d_bestX);
    cudaFree(d_bestE);
    delete[] Q;

    return 0;
}
