#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "FreeImage.h"

#define ITERATIONS 50
#define NCLUSTERS 64

/*
Execution OMP parallel:
gcc compress.c -fopenmp -O2 -lm -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o out && srun -n1 --cpus-per-task=64 --reservation=fri out
*/

int main(int argc, char *argv[])
{
    // Load image from file and convert to 32 bits
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "images/spiderman.png", 0);
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    // Image details
    int width = FreeImage_GetWidth(imageBitmap32);
    int height = FreeImage_GetHeight(imageBitmap32);
    int pitch = FreeImage_GetPitch(imageBitmap32);
    int image_size = height * pitch;
    int image_points = height * width;

    printf("IMG INFORMATION\n\tWidth: %d, Height: %d, Pitch: %d \n\n", width, height, pitch);

    // Preapare memory for a raw data copy of the image
    unsigned char *image = (unsigned char *)malloc(image_size * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(image, imageBitmap32, pitch, 32, 0xFF, 0xFF, 0xFF, TRUE);

    // Start time
    double start = omp_get_wtime();

    // Cluster points
    unsigned char clusters[NCLUSTERS][3];

    // Every cluster gets a random point of the image
    for (size_t i = 0; i < NCLUSTERS; i++)
    {
        int r = rand() % image_points;
        clusters[i][0] = image[r * 4];
        clusters[i][1] = image[r * 4 + 1];
        clusters[i][2] = image[r * 4 + 2];
    }

    // Using to store the point's cluster
    // Don't use int point_clusters[image_points] i guess.
    int *point_clusters = malloc(image_points * sizeof(int));

    for (size_t i = 0; i < ITERATIONS; i++)
    {

        // For every point find the cluster with the min euclidean distance and set the point to that cluster
        // Assign every point to the nearest cluster
        #pragma omp parallel for
        for (size_t j = 0; j < image_points; j++)
        {
            int min_euc_distance = INT32_MAX;
            // Keep track of the cluster with the min distance
            int min_cluster_index;

            // Point's rgb
            unsigned char r = image[j * 4];
            unsigned char g = image[j * 4 + 1];
            unsigned char b = image[j * 4 + 2];

            for (size_t c = 0; c < NCLUSTERS; c++)
            {
                // Cluster's rgb
                unsigned char rc = clusters[c][0];
                unsigned char gc = clusters[c][1];
                unsigned char bc = clusters[c][2];

                // No need to calculate sqrt
                int euc_distance = (r - rc) * (r - rc) + (g - gc) * (g - gc) + (b - bc) * (b - bc);

                if (euc_distance < min_euc_distance)
                {
                    min_euc_distance = euc_distance;
                    min_cluster_index = c;
                }
            }
            // Assign point to the cluster with the min distance
            point_clusters[j] = min_cluster_index;
        }

        // Calculate average for every cluster
        #pragma omp parallel for
        for (size_t j = 0; j < NCLUSTERS; j++)
        {
            // Sum for every point
            int sum[3] = {0, 0, 0};
            int n = 0;

            for (size_t p = 0; p < image_points; p++)
            {
                // For every point in the cluster
                if (point_clusters[p] == j)
                {
                    sum[0] += image[p * 4];
                    sum[1] += image[p * 4 + 1];
                    sum[2] += image[p * 4 + 2];
                    n++;
                }
            }

            // If the cluster is empty give it a random point
            if (n == 0)
            {
                int randm = (rand() % image_points) * 4;
                sum[0] += image[randm];
                sum[1] += image[randm + 1];
                sum[2] += image[randm + 2];
                n = 1;
            }
            clusters[j][0] = sum[0] / n;
            clusters[j][1] = sum[1] / n;
            clusters[j][2] = sum[2] / n;
        }
        printf("Iteration %d done.\n", i + 1);
    }

    // End time
    double time = omp_get_wtime() - start;

    printf("Finally, applying the avg from NCLUSTERS to the image \n");

    for (size_t i = 0; i < image_points; i++)
    {
        // To point its approriate cluster average
        image[i * 4] = clusters[point_clusters[i]][0];
        image[i * 4 + 1] = clusters[point_clusters[i]][1];
        image[i * 4 + 2] = clusters[point_clusters[i]][2];
    }

    // Saving
    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image, width, height, pitch,
                                                 32, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_PNG, dst, "img-out.png", 0);

    printf("\nImage is saved, freeing up memory.\n");
    printf("%d : %d image took %f seconds.\n", width, height, time);
    free(image);
    free(point_clusters);

    return 0;
}
