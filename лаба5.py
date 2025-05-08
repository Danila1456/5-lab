from itertools import permutations
import numpy as np
import timeit

def generate_permutations_algorithmic(A):
    n = A.shape[0] // 2
    E, B, C, D = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    submatrices_top = [E, B]
    permutations_list = []

    # Перебор только верхних двух подматриц
    for i in range(2):
        for j in range(2):
            if i != j:  # уникальные позиции
                new_A = np.zeros_like(A)
                new_A[:n, :n] = submatrices_top[i]
                new_A[:n, n:] = submatrices_top[j]
                new_A[n:, :n] = C  # фиксированные нижние
                new_A[n:, n:] = D
                permutations_list.append(new_A)
                print("Алгоритмический подход - новая матрица:")
                print(new_A)
                print()
    return permutations_list


def generate_permutations_python(A):
    n = A.shape[0] // 2
    E, B, C, D = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    submatrices_top = [E, B]
    permutations_list = []

    # Используем permutations для верхней части
    for perm in permutations([0, 1]):
        new_A = np.zeros_like(A)
        new_A[:n, :n] = submatrices_top[perm[0]]
        new_A[:n, n:] = submatrices_top[perm[1]]
        new_A[n:, :n] = C
        new_A[n:, n:] = D
        permutations_list.append(new_A)
        print("Подход с использованием функций Python - новая матрица:")
        print(new_A)
        print()
    return permutations_list


def optimize_matrix_with_max_weighted_sum(A):
    """
    Найти матрицу с максимальным взвешенным значением суммы подматриц,
    перебирая только верхние две подматрицы.
    """
    n = A.shape[0] // 2
    E, B, C, D = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    submatrices_top = [E, B]

    max_weighted_sum = float('-inf')
    optimal_matrix = None

    # Только перестановки верхней строки
    for perm in permutations([0, 1]):
        new_A = np.zeros_like(A)
        new_A[:n, :n] = submatrices_top[perm[0]]
        new_A[:n, n:] = submatrices_top[perm[1]]
        new_A[n:, :n] = C
        new_A[n:, n:] = D

        # Считаем суммы только четырёх подматриц
        sum_E = np.sum(new_A[:n, :n])
        sum_B = np.sum(new_A[:n, n:])
        sum_C = np.sum(new_A[n:, :n])
        sum_D = np.sum(new_A[n:, n:])

        # Взвешенные суммы: веса 1, 2, 3, 4 для E, B, C, D соответственно
        weighted_sums = [
            1 * sum_E,
            2 * sum_B,
            3 * sum_C,
            4 * sum_D
        ]
        current_weighted_sum = max(weighted_sums)

        if current_weighted_sum > max_weighted_sum:
            max_weighted_sum = current_weighted_sum
            optimal_matrix = new_A.copy()

        print(f"Перестановка верхних подматриц: {perm}")
        print(f"Новая матрица:\n{new_A}")
        print(f"Суммы подматриц: E={sum_E}, B={sum_B}, C={sum_C}, D={sum_D}")
        print(f"Взвешенные суммы: {weighted_sums}")
        print(f"Максимальное взвешенное значение: {current_weighted_sum}")
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

    # Измерение времени выполнения
    def run_algorithmic():
        return generate_permutations_algorithmic(A)

    def run_python():
        return generate_permutations_python(A)

    def run_optimize():
        return optimize_matrix_with_max_weighted_sum(A)

    algorithmic_time = timeit.timeit(run_algorithmic, number=1)
    print(f"Алгоритмический подход: {algorithmic_time:.6f} секунд")

    python_time = timeit.timeit(run_python, number=1)
    print(f"Подход с использованием функций Python: {python_time:.6f} секунд")

    optimize_time = timeit.timeit(run_optimize, number=1)
    print(f"Оптимизация матрицы: {optimize_time:.6f} секунд")
