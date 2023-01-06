#include "k_means.h"


#define NUM_THREADS_PER_BLOCK 256

using namespace std;

typedef struct point{
	float x;
	float y;
} point;

int NUM_BLOCKS;
int N;								//Nr of total Points generated
int K;								//Nr of total clusters

struct point* points;				//Array of generated points
struct point *centroides_antigos;	//Previous centroids
int* cluster_size;
float* sum_cluster_x;
float* sum_cluster_y;
struct point* cluster_centroid;


//Calculates the centroide of a cluster
__device__
void calcular_centroideKernel(int K, point *cluster_centroid, float *sum_cluster_x, float *sum_cluster_y, int *cluster_size){

	for (int k = 0; k < K; k++) {
		//ads centroid and total points nr to global cluster
		cluster_centroid[k].x = sum_cluster_x[k]/cluster_size[k];
		cluster_centroid[k].y = sum_cluster_y[k]/cluster_size[k];
	}
	
}

//Euclidian distance
//Note: By removing the sqr calculation we get a much faster code that,
//in the end, gives the same outcome (The closest centroide for each point)
__device__
float distancia_euclidianaKernel(point a, point b){
	return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}

//Compares the old centroid with the new one to check if it has changed
//If it has changed sets result to 0 otherwise sets to 1
__global__ 
void comparar_centroidesKernel(int K, point *cluster_centroid, point *centroides_antigos, int *result) {

    result[0] = 0;

    for(int i = 0; i < K; i++){
        if(cluster_centroid[i].x != centroides_antigos[i].x || cluster_centroid[i].y != centroides_antigos[i].y) {
            result[0] = 1;
        }
    }
}

//Function that designates a point to the closest cluster with cuda
__global__
void atribuir_clustersKernel(int k,int n, point *points, int *cluster_size, float *sum_cluster_x, float *sum_cluster_y, point *cluster_centroid, point *centroides_antigos) {
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	if(threadID < n) {
		int cluster_mais_proximo = 0;
		point cent = cluster_centroid[0], p = points[threadID];
		float menor_distancia = distancia_euclidianaKernel(p,cent);


		for (int j = 1; j < k; j++){
			float distancia = distancia_euclidianaKernel(p,cluster_centroid[j]);

				if(distancia < menor_distancia){
					cluster_mais_proximo = j;
					menor_distancia = distancia;
				}		
		}

		
		atomicAdd(&sum_cluster_x[cluster_mais_proximo],p.x);
		atomicAdd(&sum_cluster_y[cluster_mais_proximo],p.y);
		atomicAdd(&cluster_size[cluster_mais_proximo],1);
	}
	
	
}

//Creates N Random points and assigns the first K points as centroids of each cluster
void inicializa() {
	points = (point*) malloc(N * sizeof(point));
	cluster_size = (int*) malloc(K * sizeof(int));
	sum_cluster_x = (float*) malloc(K * sizeof(float));
	sum_cluster_y = (float*) malloc(K * sizeof(float));
	cluster_centroid = (point*) malloc(K * sizeof(point));
	centroides_antigos = (point*) malloc(K * sizeof(point));


	
	srand(10);
	for(int i = 0; i < N; i++) {
		points[i].x = (float) rand() / RAND_MAX;
		points[i].y = (float) rand() / RAND_MAX;
	}

	for(int i = 0; i < K; i++) {
		cluster_centroid[i].x = points[i].x;
		cluster_centroid[i].y = points[i].y;
		sum_cluster_x[i] = 0;
		sum_cluster_y[i] = 0;	
		cluster_size[i] = 0; 
	}

}



//Clears memory allocated for the arrays
void freeMemory(){
		free(points);
		free(centroides_antigos);
		free(cluster_size);
		free(sum_cluster_x);
		free(sum_cluster_y);
		free(cluster_centroid);
}

void launchKernel (){
	//pointers to the device memory
    point *points_device; 
    int *cluster_size_device;
    float *sum_cluster_x_device, *sum_cluster_y_device;
    point *cluster_centroid_device,*centroides_antigos_device;
	int *ended_device;

    // declare variable with size of the array in bytes
    int points_bytes = N * sizeof(point);
    int cluster_size_bytes = K * sizeof(int);
    int sum_bytes = K * sizeof(float);
    int cluster_centroid_bytes = K * sizeof(point);


    // allocate the memory on the device
    cudaMalloc((void**)&points_device, points_bytes);
    cudaMalloc((void**)&cluster_size_device, cluster_size_bytes);
    cudaMalloc((void**)&sum_cluster_x_device, sum_bytes);
    cudaMalloc((void**)&sum_cluster_y_device, sum_bytes);
    cudaMalloc((void**)&cluster_centroid_device, cluster_centroid_bytes);
    cudaMalloc((void**)&centroides_antigos_device, cluster_centroid_bytes);
	cudaMalloc((void**)&ended_device, sizeof(int));
    checkCUDAError("mem allocation");

    // Initialize the device memory for cluster_size_device, sum_cluster_x_device, sum_cluster_y_device, centroides_antigos_device to 0 asynchronously
    cudaMemsetAsync(cluster_size_device, 0, cluster_size_bytes, cudaStreamDefault);
    cudaMemsetAsync(sum_cluster_x_device, 0, sum_bytes, cudaStreamDefault);
    cudaMemsetAsync(sum_cluster_y_device, 0, sum_bytes, cudaStreamDefault);
    cudaMemsetAsync(centroides_antigos_device, 0, cluster_centroid_bytes, cudaStreamDefault);

    //copy inputs to the device asynchronously
    cudaMemcpyAsync(points_device, points, points_bytes, cudaMemcpyHostToDevice, cudaStreamDefault);
    cudaMemcpyAsync(cluster_centroid_device, cluster_centroid, cluster_centroid_bytes, cudaMemcpyHostToDevice, cudaStreamDefault);

    // Launch the kernel
    int iterations = 0;
    int ended = 0;

    startKernelTime();
	while (!ended && iterations < 20) {
        atribuir_clustersKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStreamDefault>>>(K, N, points_device, cluster_size_device, sum_cluster_x_device, sum_cluster_y_device, cluster_centroid_device, centroides_antigos_device);
        iterations++;
        if (iterations < 20)
		comparar_centroidesKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(K, cluster_centroid_device, centroides_antigos_device, ended_device);
		// Copy the result from device memory to host memory
    	cudaMemcpy(&ended, ended_device, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Synchronize the streams to ensure all the kernel launches have completed
    cudaDeviceSynchronize();

    // Stop the timer
    stopKernelTime();

    // Copy the required output to the host asynchronously
    cudaMemcpyAsync(cluster_centroid, cluster_centroid_device, cluster_centroid_bytes, cudaMemcpyDeviceToHost, cudaStreamDefault);
    cudaMemcpyAsync(cluster_size, cluster_size_device, cluster_size_bytes, cudaMemcpyDeviceToHost, cudaStreamDefault);

    // Synchronize the streams to ensure the data has been transferred
    cudaDeviceSynchronize();

    // Free the device memory
    cudaFree(points_device);
    cudaFree(cluster_size_device);
    cudaFree(sum_cluster_x_device);
    cudaFree(sum_cluster_y_device);
    cudaFree(cluster_centroid_device);
    cudaFree(centroides_antigos_device);
	cudaFree(ended_device);

    // Free the host memory
    cudaFreeHost(points);
    cudaFreeHost(cluster_size);
    cudaFreeHost(sum_cluster_x);
    cudaFreeHost(sum_cluster_y);
    cudaFreeHost(cluster_centroid);
    cudaFreeHost(centroides_antigos);

    // Reset the device
    cudaDeviceReset();

    // Print the output
    printf("N = %d, K = %d\n", N, K);

    for (int k = 0; k < K; k++) {
    printf("Center: (%0.3f, %0.3f) : Size: %d\n", cluster_centroid[k].x, cluster_centroid[k].y, cluster_size[k]);
    }
    printf("Iterations: %d\n", iterations);
}

int main(int argc, char *argv[]){

	if (argc < 3) {
		printf("Not enough arguments!");
		return -1;
	}

	//N = atoi(argv[1]);
	N = atoi(argv[1]);
	K = atoi(argv[2]);
	NUM_BLOCKS = N / NUM_THREADS_PER_BLOCK + 1;
	

	inicializa();


	launchKernel();

	freeMemory();

	return 0;
}
