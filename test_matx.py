import matx
import timeit

def fib(n: int) -> int:
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


if __name__ == '__main__':
    fib_script = matx.script(fib)
    
    # test on Macbook with m1 chip
    print(f'Python execution time: {timeit.timeit(lambda: fib(30), number=10)}s')  # 1.59s
    print(f'Matx execution time: {timeit.timeit(lambda: fib_script(30), number=10)}s') # 0.03s