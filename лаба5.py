from itertools import permutations
import numpy as np
import timeit

def generate_permutations_algorithmic(A):
    n = A.shape[0] // 2
    E, B, C, D = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    permutations_list = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    if len(set([i, j, k, l])) == 4:
                        submatrices = [E, B, C, D]
                        new_A = np.zeros_like(A)
                        new_A[:n, :n] = submatrices[i]
                        new_A[:n, n:] = submatrices[j]
                        new_A[n:, :n] = submatrices[k]
                        new_A[n:, n:] = submatrices[l]
                        permutations_list.append(new_A)
                        print("Алгоритмический подход - новая матрица:")
                        print(new_A)
                        print()
    return permutations_list


def generate_permutations_python(A):
    n = A.shape[0] // 2
    E, B, C, D = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    submatrices = [E, B, C, D]
    permutations_list = []
    for perm in permutations(range(4)):
        new_A = np.zeros_like(A)
        new_A[:n, :n] = submatrices[perm[0]]
        new_A[:n, n:] = submatrices[perm[1]]
        new_A[n:, :n] = submatrices[perm[2]]
        new_A[n:, n:] = submatrices[perm[3]]
        permutations_list.append(new_A)
        print("Подход с использованием функций Python - новая матрица:")
        print(new_A)
        print()
    return permutations_list


def optimize_matrix_with_max_weighted_sum(A):
    """
    Найти матрицу с максимальным взвешенным значением суммы подматриц.
    """
    n = A.shape[0] // 2
    E, B, C, D = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    submatrices = [E, B, C, D]

    max_weighted_sum = float('-inf')
    optimal_matrix = None

    for perm in permutations(range(4)):
        # Создаем новую матрицу на основе текущей перестановки
        new_A = np.zeros_like(A)
        new_A[:n, :n] = submatrices[perm[0]]
        new_A[:n, n:] = submatrices[perm[1]]
        new_A[n:, :n] = submatrices[perm[2]]
        new_A[n:, n:] = submatrices[perm[3]]

        # Вычисляем сумму каждой подматрицы
        sums = [np.sum(submatrices[perm[i]]) for i in range(4)]

        # Вычисляем взвешенные суммы (сумма * номер подматрицы)
        weighted_sums = [(i + 1) * sums[i] for i in range(4)]

        # Находим максимальное взвешенное значение для текущей перестановки
        current_max_weighted_sum = max(weighted_sums)

        # Если текущее значение больше максимального, обновляем результат
        if current_max_weighted_sum > max_weighted_sum:
            max_weighted_sum = current_max_weighted_sum
            optimal_matrix = new_A.copy()

        print(f"Перестановка: {perm}")
        print(f"Новая матрица:\n{new_A}")
        print(f"Суммы подматриц: {sums}")
        print(f"Взвешенные суммы: {weighted_sums}")
        print(f"Максимальное взвешенное значение: {current_max_weighted_sum}")
        print()

    print("Оптимальная матрица с максимальным взвешенным значением:")
    print(optimal_matrix)
    print(f"Максимальное взвешенное значение: {max_weighted_sum}")

    return optimal_matrix, max_weighted_sum

if __name__ == "__main__":
    # Исходная матрица
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

    print("Исходная матрица:")
    print(A)
    print()

    # Измерение времени выполнения с помощью timeit
    def run_algorithmic():
        return generate_permutations_algorithmic(A)

    def run_python():
        return generate_permutations_python(A)

    def run_optimize():
        return optimize_matrix_with_max_weighted_sum(A)

    # Алгоритмический подход
    algorithmic_time = timeit.timeit(run_algorithmic, number=1)
    print(f"Алгоритмический подход: {algorithmic_time:.6f} секунд")
# Подход с использованием функций Python
    python_time = timeit.timeit(run_python, number=1)
    print(f"Подход с использованием функций Python: {python_time:.6f} секунд")

    # Оптимизация матрицы (новое усложнение)
    optimize_time = timeit.timeit(run_optimize, number=1)
    print(f"Оптимизация матрицы: {optimize_time:.6f} секунд")