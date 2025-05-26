import pandas as pd
import numpy as np
import rasterio
from numba import cuda
import math
import sys

@cuda.jit
def algo(A, B, results, kernel):
    x, y = cuda.grid(2)

    if x < A.shape[1] and y < A.shape[2]:
        temp_data = cuda.local.array((7, 31, 31), dtype=np.float32)   # maybe replace by shared array?
        temp_kernel = cuda.local.array((31, 31), dtype=np.float32)

        # initialize temp_data to zero
        for i in range(7):
            for j in range(31):
                for k in range(31):
                    temp_data[i, j, k] = 0.0

        # initialize temp_kernel to zero
        for i in range(31):
            for j in range(31):
                temp_kernel[i, j] = 0.0

        if A[-1, x, y] != -9999.0:
            pivot_pix = int(A[-1, x, y])
            results[pivot_pix, 0] = pivot_pix

            # cut the bbox
            rmin = max(0, x - kernel.shape[1] // 2)
            rmax = min(A.shape[1], x + kernel.shape[1] // 2 + 1)
            cmin = max(0, y - kernel.shape[0] // 2)
            cmax = min(A.shape[2], y + kernel.shape[0] // 2 + 1)

            # iterate over the cut bbox and drop pixels not in isochrone
            for i in range(rmin, rmax):
                for j in range(cmin, cmax):
                    # find the neighboring pixels to include from indexing_arr
                    found = False
                    for b_idx in range(B.shape[1]):
                        if A[-2, i, j] == B[pivot_pix, b_idx]:
                            found = True
                            break

                    if found and A[-2, i, j] != -9999.0:
                        temp_kernel[i - rmin, j - cmin] = kernel[i - rmin, j - cmin]
                        for b in range(7):  # assuming 7 bands
                            temp_data[b, i - rmin, j - cmin] = A[b, i, j]
                    else:
                        for b in range(7):
                            temp_data[b, i - rmin, j - cmin] = -9999.0

            # compute the sum of the kernel weights
            kernel_sum = 0.0
            for ki in range(temp_kernel.shape[0]):
                for kj in range(temp_kernel.shape[1]):
                    kernel_sum += temp_kernel[ki, kj]

            # normalize the kernel weights to add up to 1 if the sum is not zero
            if kernel_sum > 0:
                for ki in range(temp_kernel.shape[0]):
                    for kj in range(temp_kernel.shape[1]):
                        temp_kernel[ki, kj] /= kernel_sum

            for b in range(temp_data.shape[0]):
                sum_result = 0.0  # initialize sum for this band
                for ti in range(temp_data.shape[1]):
                    for tj in range(temp_data.shape[2]):
                        sum_result += temp_data[b, ti, tj] * temp_kernel[ti, tj]

                results[pivot_pix, b+1] = sum_result


def create_gaussian_kernel(size, sigma):
    # calculate the center index
    center = size // 2

    # initialize the kernel
    kernel = np.zeros((size, size))

    # compute the Gaussian function
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # normalize the kernel so that the sum of all elements is 1
    kernel /= np.sum(kernel)

    return kernel


def process_images(A, B, kernel_size, sigma):
    # define device arrays
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)

    results = np.zeros(((A.shape[1]*A.shape[2]), 8), dtype=np.float32)
    d_results = cuda.to_device(results)

    # generate a kernel
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    d_kernel = cuda.to_device(kernel)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(A.shape[1] / threadsperblock[0])
    blockspergrid_y = math.ceil(A.shape[2] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    algo[blockspergrid, threadsperblock](d_A, d_B, d_results, d_kernel)

    cuda.synchronize()  # ensuring GPU has finished processing
    result_host = d_results.copy_to_host()

    # explicitly clean up device memory
    del d_A, d_B, d_results

    return result_host


def main(grid_num):
    # load the image data
    with rasterio.open(f'data/iso_for_gpu/img_{grid_num}.tif') as src:
        image = src.read()

    image[:7][np.isnan(image[:7])] = 0
    image[7:][np.isnan(image[7:])] = -9999.0
    print(image.shape)

    # load the indexing array
    loaded = np.load(f'data/iso_for_gpu/indexing_arr_{grid_num}.npz')
    indexing_arr = loaded['array']
    indexing_arr[np.isnan(indexing_arr)] = -9999.0
    print(indexing_arr.shape)

    results = process_images(image, indexing_arr, kernel_size=31, sigma=7)
    print(len(results))
    print(f'Processed results for grid {grid_num}')
    print()
    
    # convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(results, columns=['index', 'street_walk_length', 'num_street_intersections', 'ndvi', 'ent_5',
          'slope', 'population', 'pub_trans_count'])

    # save the DataFrame to a CSV file
    csv_file_path = f'iso_for_gpu/processed_data/processed_data_{grid_num}.csv'
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    grid_num = sys.argv[1]
    main(grid_num)