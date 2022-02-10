#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "FreeImage.h"
#include <omp.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE 16384
#define NCLUSTERS 64
#define ITERATIONS 50

/*
Execution GPU:
module load CUDA
gcc compressGPU.c -fopenmp -O2 -lm -lOpenCL -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o out && srun -n1 -G1 --reservation=fri out
*/

int main(int argc, char const *argv[])
{
    // Load image from file and convert to 32 bits
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "images/3840.png", 0);
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    // Image details
    int width = FreeImage_GetWidth(imageBitmap32);
    int height = FreeImage_GetHeight(imageBitmap32);
    int pitch = FreeImage_GetPitch(imageBitmap32);
    int image_size = height * pitch;
    int image_points = height * width;

    unsigned char *image = (unsigned char *)malloc(image_size * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(image, imageBitmap32, pitch, 32, 0xFF, 0xFF, 0xFF, TRUE);

    // Clusters[NCLUSTERS][3] RGB RGB
    // clusters_sum[NCLUSTERS][3] sum(r) sum(b) sum(g)
    // clusters_n[NCLUSTERS] pointCusters[image_points]
    unsigned char *clusters = (unsigned char *)malloc((NCLUSTERS * 3) * sizeof(unsigned char));
    unsigned int *clusters_sum = (unsigned int *)malloc((NCLUSTERS * 3) * sizeof(unsigned int));
    unsigned int *clusters_n = (unsigned int *)malloc(NCLUSTERS * sizeof(unsigned int));
    unsigned int *point_clusters = (unsigned int *)malloc(image_points * sizeof(unsigned int));

    // Used for a random number generator
    ulong seed = 123456789;

    for (size_t i = 0; i < NCLUSTERS; i++)
    {
        int r = rand() % image_points;
        clusters[i * 3] = image[r * 4];
        clusters[i * 3 + 1] = image[r * 4 + 1];
        clusters[i * 3 + 2] = image[r * 4 + 2];

        clusters_sum[i * 3] = 0;
        clusters_sum[i * 3 + 1] = 0;
        clusters_sum[i * 3 + 2] = 0;
        clusters_n[i] = 0;
    }

    cl_int ret;
    // Branje datoteke
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("kernel_local.cl", "r");
    if (!fp)
    {
        fprintf(stderr, ":-(#\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    // Podatki o platformi
    cl_platform_id platform_id[10];
    cl_uint ret_num_platforms;
    char *buf;
    size_t buf_len;
    ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
    // max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform

    // Podatki o napravi
    cl_device_id device_id[10];
    cl_uint ret_num_devices;

    // Delali bomo s platform_id[0] na GPU
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
                         device_id, &ret_num_devices);
    // izbrana platforma, tip naprave, koliko naprav nas zanima
    // kazalec na naprave, dejansko "stevilo naprav

    // Kontekst
    cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);
    // kontekst: vklju"cene platforme - NULL je privzeta, "stevilo naprav,
    // kazalci na naprave, kazalec na call-back funkcijo v primeru napake
    // dodatni parametri funkcije, "stevilka napake

    // Ukazna vrsta
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
    // kontekst, naprava, INORDER/OUTOFORDER, napake

    // Alokacija pomnilnika na napravi
    cl_mem image_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, image_size, image, &ret);
    cl_mem clusters_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (NCLUSTERS * 3) * sizeof(unsigned char), clusters, &ret);
    cl_mem clusters_sum_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (NCLUSTERS * 3) * sizeof(unsigned int), clusters_sum, &ret);
    cl_mem clusters_n_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NCLUSTERS * sizeof(unsigned int), clusters_n, &ret);
    cl_mem point_clusters_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, image_points * sizeof(unsigned int), point_clusters, &ret);

    // Priprava programa
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                                   NULL, &ret);
    // kontekst, "stevilo kazalcev na kodo, kazalci na kodo,
    // stringi so NULL terminated, napaka

    // Prevajanje
    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
    // program, "stevilo naprav, lista naprav, opcije pri prevajanju,
    // kazalec na funkcijo, uporabni"ski argumenti

    // Log
    size_t build_log_len;
    char *build_log;
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                0, NULL, &build_log_len);
    // program, "naprava, tip izpisa,
    // maksimalna dol"zina niza, kazalec na niz, dejanska dol"zina niza
    build_log = (char *)malloc(sizeof(char) * (build_log_len + 1));
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);

    // "s"cepec: priprava objekta
    cl_kernel kernel1 = clCreateKernel(program, "Min_euc_dist", &ret);
    cl_kernel kernel2 = clCreateKernel(program, "Set_avg", &ret);
    // cl_kernel kernel3 = clCreateKernel(program, "reset", &ret);
    // program, ime "s"cepca, napaka

    // "s"cepec: argumenti
    ret = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&image_mem_obj);
    ret |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&clusters_mem_obj);
    ret |= clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&clusters_sum_mem_obj);
    ret |= clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&clusters_n_mem_obj);
    ret |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void *)&point_clusters_mem_obj);
    ret |= clSetKernelArg(kernel1, 5, sizeof(cl_int), (void *)&image_points);
    // "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

    ret = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&clusters_mem_obj);
    ret |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&clusters_sum_mem_obj);
    ret |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&clusters_n_mem_obj);
    ret |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void *)&image_mem_obj);
    ret |= clSetKernelArg(kernel2, 4, sizeof(cl_int), (void *)&image_points);
    ret |= clSetKernelArg(kernel2, 5, sizeof(cl_ulong), (void *)&seed);


    // Delitev dela - Dividing the workload 256
    size_t local_item_size1 = 256;
    size_t num_groups = ((image_points - 1) / local_item_size1 + 1);
    size_t global_item_size1 = num_groups * local_item_size1;

    printf("%d groups with %d local itemsize and %d global itemsize.\n", num_groups, local_item_size1, global_item_size1);

    printf("%d %d\n", local_item_size1, global_item_size1);

    size_t local_item_size2 = 64;
    size_t global_item_size2 = 256;

    // Time for the GPU
    double start = omp_get_wtime();
    // "s"cepec: zagon

    for (size_t i = 0; i < ITERATIONS; i++)
    {
        printf("Doing iteration %d\n", i);

        ret = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL,
                                     &global_item_size1, &local_item_size1, 0, NULL, NULL);

        ret = clEnqueueNDRangeKernel(command_queue, kernel2, 1, NULL,
                                     &global_item_size2, &local_item_size2, 0, NULL, NULL);

        // ret = clEnqueueNDRangeKernel(command_queue, kernel3, 1, NULL,
        //                              &global_item_size2, &local_item_size2, 0, NULL, NULL);
        // // vrsta, "s"cepec, dimenzionalnost, mora biti NULL,
        // kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti,
        // dogodki, ki se morajo zgoditi pred klicem
    }

    // Kopiranje rezultatov
    // ret = clEnqueueReadBuffer(command_queue, image_mem_obj, CL_TRUE, 0,
    //                           image_size, image, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, clusters_mem_obj, CL_TRUE, 0,
                              (NCLUSTERS * 3) * sizeof(unsigned char), clusters, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, point_clusters_mem_obj, CL_TRUE, 0,
                              image_points * sizeof(unsigned int), point_clusters, 0, NULL, NULL);
    // branje v pomnilnik iz naparave, 0 = offset
    // zadnji trije - dogodki, ki se morajo zgoditi prej

    // End timer
    double time = omp_get_wtime() - start;

    printf("Printing the read cluster\n");
    for (size_t i = 0; i < image_points; i++)
    {
        // printf("%d ", clusters[point_clusters[i] + 0]);
        // To point its approriate cluster average
        image[i * 4] = clusters[point_clusters[i] * 3];
        image[i * 4 + 1] = clusters[point_clusters[i] * 3 + 1];
        image[i * 4 + 2] = clusters[point_clusters[i] * 3 + 2];
    }

    // Saving
    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image, width, height, pitch,
                                                 32, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_PNG, dst, "image_out.png", 0);

    printf("\n\nImage is saved, freeing up memory\n");
    printf("%d : %d image took %f seconds\n", width, height, time);

    free(image);
    free(clusters);
    free(clusters_sum);
    free(clusters_n);
    free(point_clusters);

    // "ci"s"cenje
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel1);
    ret = clReleaseKernel(kernel2);
    ret = clReleaseProgram(program);

    ret = clReleaseMemObject(image_mem_obj);
    ret = clReleaseMemObject(clusters_mem_obj);
    ret = clReleaseMemObject(clusters_sum_mem_obj);
    ret = clReleaseMemObject(clusters_n_mem_obj);
    ret = clReleaseMemObject(point_clusters_mem_obj);

    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}


