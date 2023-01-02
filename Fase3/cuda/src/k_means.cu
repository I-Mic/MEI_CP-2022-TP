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
//If it has changed returns 0 otherwise returns 1
__device__
int comparar_centroidesKernel(int K,point *cluster_centroid, point *centroides_antigos){
	for(int i = 0; i < K; i++){
		if(cluster_centroid[i].x != centroides_antigos[i].x || cluster_centroid[i].y != centroides_antigos[i].y) return 0;
	}
	return 1;
}

//Resets the clusters, saving the current centroids, for the next iteration
__device__
void reset_clustersKernel(int K,int *cluster_size, float *sum_cluster_x, float *sum_cluster_y,point *cluster_centroid, point *centroides_antigos){
	for (int k = 0; k < K; k++){
		centroides_antigos[k] = cluster_centroid[k]; 
		sum_cluster_x[k] = 0;
		sum_cluster_y[k] = 0;
		cluster_size[k] = 0;

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

__global__
void reset_IterationKernel(int k, int *cluster_size, float *sum_cluster_x, float *sum_cluster_y, point *cluster_centroid, point *centroides_antigos, int ended){
	if(threadIdx.x == 0 && blockIdx.x == 0){
		calcular_centroideKernel(k, cluster_centroid, sum_cluster_x, sum_cluster_y, cluster_size);
		ended = comparar_centroidesKernel(k,cluster_centroid, centroides_antigos);
		reset_clustersKernel(k,cluster_size, sum_cluster_x, sum_cluster_y,cluster_centroid, centroides_antigos);
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
	point *pointsa; 
	int *cluster_sizea;
	float *sum_cluster_xa, *sum_cluster_ya;
	point *cluster_centroida,*centroides_antigosa;

	// declare variable with size of the array in bytes
	int points_Bytes = N * sizeof(point);
	int cluster_size_Bytes = K * sizeof(int);
	int sum_Bytes = K * sizeof(float);
	int cluster_centroid_Bytes = K * sizeof(point);


	// allocate the memory on the device
	cudaMalloc((void**)&pointsa, points_Bytes);
	cudaMalloc((void**)&cluster_sizea, cluster_size_Bytes);
	cudaMalloc((void**)&sum_cluster_xa, sum_Bytes);
	cudaMalloc((void**)&sum_cluster_ya, sum_Bytes);
	cudaMalloc((void**)&cluster_centroida, cluster_centroid_Bytes);
	cudaMalloc((void**)&centroides_antigosa, cluster_centroid_Bytes);
	checkCUDAError("mem allocation");

	//copy inputs to the device
	cudaMemcpy(pointsa, points, points_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(cluster_sizea, cluster_size, cluster_size_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(sum_cluster_xa, sum_cluster_x, sum_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(sum_cluster_ya, sum_cluster_y, sum_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(cluster_centroida, cluster_centroid, cluster_centroid_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(centroides_antigosa, centroides_antigos, cluster_centroid_Bytes, cudaMemcpyHostToDevice);
	checkCUDAError("memcpy h->d");

	//lauch the Kernel
	int iteracoes = 0;
	int ended = 0;

	startKernelTime ();
	while(!ended && iteracoes < 20) {
		atribuir_clustersKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(K,N,pointsa,cluster_sizea,sum_cluster_xa,sum_cluster_ya,cluster_centroida,centroides_antigosa);
		iteracoes++;
		if(iteracoes < 20)
		reset_IterationKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(K,cluster_sizea,sum_cluster_xa, sum_cluster_ya, cluster_centroida, centroides_antigosa, ended);
	}
	stopKernelTime ();
	checkCUDAError("kernel invocation");
	

	//copy the required output to the host
	cudaMemcpy(cluster_centroid, cluster_centroida, cluster_centroid_Bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cluster_size, cluster_sizea, cluster_size_Bytes, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy h->d");

	//free the device memory
	cudaFree(pointsa);
	cudaFree(cluster_sizea);
	cudaFree(sum_cluster_xa);
	cudaFree(sum_cluster_ya);
	cudaFree(cluster_centroida);
	cudaFree(centroides_antigosa);
	checkCUDAError("mem free");

	//print the output
	printf("N = %d, K = %d\n",N,K);

	for(int k = 0; k < K; k++){
		printf("Center: (%0.3f, %0.3f) : Size: %d\n",cluster_centroid[k].x,cluster_centroid[k].y,cluster_size[k]);
	}
	printf("Iterations:%d\n",iteracoes);

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
