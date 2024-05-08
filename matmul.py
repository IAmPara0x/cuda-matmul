#!/usr/bin/env python

import numpy as np

def save_mat(filename, mat):

    with open(filename, "w") as f:

        # assume 2 dim matrix
        row,col = mat.shape


        for i in range(row):

            row_str = ""
            for j in range(col):

                if j == col - 1:
                    row_str = row_str + str(f"{mat[i][j]}")
                else:
                    row_str = row_str + str(f"{mat[i][j]},")

            row_str = row_str + "\n"
            f.write(row_str)

if __name__ == "__main__":

    # pytorch: 2.74 ms ± 7.56 µs
    SIZE=1024
    print(f"{SIZE}")
    A = np.random.randn(SIZE, SIZE).astype(np.float16)
    B = np.random.randn(SIZE, SIZE).astype(np.float16)
    C = A @ B

    save_mat("A.txt", A)
    save_mat("B.txt", B)
    save_mat("C.txt", C)
