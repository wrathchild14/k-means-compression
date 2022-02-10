#define NCLUSTERS 64

__kernel void Min_euc_dist(__global unsigned char *image,
                           __global unsigned char *clusters,
                           __global unsigned int *clusters_sum,
                           __global unsigned int *clusters_n,
                           __global unsigned int *point_clusters,
                           int image_points) {
  int i_global = get_global_id(0);
  int i_local = get_local_id(0);

  // Just save the cluster into a local array
  __local int clusters_local[3 * NCLUSTERS];

  if (i_global < image_points) {
    if (i_local < NCLUSTERS) {
      clusters_local[i_local * 3] = clusters[i_local * 3];
      clusters_local[i_local * 3 + 1] = clusters[i_local * 3 + 1];
      clusters_local[i_local * 3 + 2] = clusters[i_local * 3 + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int min_euc_distance = INT_MAX;
    int min_cluster_index;

    unsigned char r = image[i_global * 4];
    unsigned char g = image[i_global * 4 + 1];
    unsigned char b = image[i_global * 4 + 2];

    for (int c = 0; c < NCLUSTERS; c++) {
      // Cluster's rgb
      unsigned char rc = clusters_local[c * 3];
      unsigned char gc = clusters_local[c * 3 + 1];
      unsigned char bc = clusters_local[c * 3 + 2];

      int euc_distance =
          (r - rc) * (r - rc) + (g - gc) * (g - gc) + (b - bc) * (b - bc);

      if (euc_distance < min_euc_distance) {
        min_euc_distance = euc_distance;
        min_cluster_index = c;
      }
    }
    // Assign point to the cluster with the min distance
    point_clusters[i_global] = min_cluster_index;

    atomic_add(&clusters_sum[min_cluster_index * 3], r);
    atomic_add(&clusters_sum[min_cluster_index * 3 + 1], g);
    atomic_add(&clusters_sum[min_cluster_index * 3 + 2], b);
    atomic_inc(&clusters_n[min_cluster_index]);
  }
}

__kernel void Set_avg(__global unsigned char *clusters,
                      __global unsigned int *clusters_sum,
                      __global unsigned int *clusters_n,
                      __global unsigned char *image, int image_points,
                      ulong seedUlong) {
  int i = get_local_id(0);

  if (i < NCLUSTERS) {
    // printf("GOING THROUGHT %d CLUSTER\n", i);
    if (clusters_n[i] == 0) {

      ulong seed = seedUlong + i;
      seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
      uint result = seed >> 16;
      // printf("%d iamgepoints: %d\n", result, image_points);

      int r = (result % image_points) * 4;
      // printf("clister %d is empty and needs a point, random is %d %d %d\n",
      // i, image[r], image[r+1], image[r+2]); Often gives 0 0 0 point idk how
      // to fix.

      clusters_sum[i * 3] = image[r];
      clusters_sum[i * 3 + 1] = image[r + 1];
      clusters_sum[i * 3 + 2] = image[r + 2];
      clusters_n[i] = 3;
    }

    clusters[i * 3] = clusters_sum[i * 3] / clusters_n[i];
    clusters[i * 3 + 1] = clusters_sum[i * 3 + 1] / clusters_n[i];
    clusters[i * 3 + 2] = clusters_sum[i * 3 + 2] / clusters_n[i];

    clusters_sum[i * 3] = 0;
    clusters_sum[i * 3 + 1] = 0;
    clusters_sum[i * 3 + 2] = 0;
    clusters_n[i] = 0;
  }
}
