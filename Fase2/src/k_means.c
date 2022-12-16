#include "../include/utils.h"

typedef struct point{
	float x;
	float y;
} point;


int N;								//Nr of total Points generated
int K;								//Nr of total clusters
int T;								//Nr of total threads

struct point* points;				//Array of generated points
struct point *centroides_antigos;	//Previous centroids
int* cluster_size;
float* sum_cluster_x;
float* sum_cluster_y;
struct point* cluster_centroid;


//Calculates the centroide of a cluster
void calcular_centroide(){

	#pragma omp parallel for
	for (int k = 0; k < K; k++) {
		//ads centroid and total points nr to global cluster
		cluster_centroid[k].x = sum_cluster_x[k]/cluster_size[k];
		cluster_centroid[k].y = sum_cluster_y[k]/cluster_size[k];
	}
	
}

//Euclidian distance
//Note: By removing the sqr calculation we get a much faster code that,
//in the end, gives the same outcome (The closest centroide for each point)
float distancia_euclidiana(point a, point b){
	return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}


//Compares the old centroid with the new one to check if it has changed
//If it has changed returns 0 otherwise returns 1
int comparar_centroides(){
	for(int i = 0; i < K; i++){
		if(cluster_centroid[i].x != centroides_antigos[i].x || cluster_centroid[i].y != centroides_antigos[i].y) return 0;
	}
	return 1;
}

//Resets the clusters, saving the current centroids, for the next iteration
//Added paralelism for better performance
void reset_clusters(){
	#pragma omp parallel for
	for (int k = 0; k < K; k++){
		centroides_antigos[k] = cluster_centroid[k]; 
		sum_cluster_x[k] = 0;
		sum_cluster_y[k] = 0;
		cluster_size[k] = 0;

	}
}


//Function that designates a point to the closest cluster with paralelism
//If any cluster changed at the end returns 1, otherwise returns 0.
int atribuir_clusters() {

	#pragma omp parallel for reduction(+:cluster_size[:K]) reduction (+:sum_cluster_x[:K]) reduction(+:sum_cluster_y[:K])
	for (int i = 0; i < N; i++){
		int cluster_mais_proximo = 0;
		point cent = cluster_centroid[0], p = points[i];
		float menor_distancia = distancia_euclidiana(p,cent);

		for (int j = 1; j < K; j++){
			float distancia = distancia_euclidiana(p,cluster_centroid[j]);

				if(distancia < menor_distancia){
					cluster_mais_proximo = j;
					menor_distancia = distancia;
				}		
		}
		cluster_size[cluster_mais_proximo]++;
		sum_cluster_x[cluster_mais_proximo] += p.x;
		sum_cluster_y[cluster_mais_proximo] += p.y;	
	}

	calcular_centroide();
	return comparar_centroides();
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

	#pragma omp parallel for
	for(int i = 0; i < K; i++) {
		cluster_centroid[i].x = points[i].x;
		cluster_centroid[i].y = points[i].y;
		sum_cluster_x[i] = 0;
		sum_cluster_y[i] = 0;	
		cluster_size[i] = 0; 
	}
}


//Function that aplies Lloyd algorithm
//Only ends when the aplication has converged 
void k_means_lloyd_algorithm() {

	int iteracoes = 0;

	inicializa();

	while(!atribuir_clusters() && iteracoes < 20) {
		reset_clusters();
		iteracoes++;
	}

	printf("N = %d, K = %d\n",N,K);

	for(int k = 0; k < K; k++){
		printf("Center: (%0.3f, %0.3f) : Size: %d\n",cluster_centroid[k].x,cluster_centroid[k].y,cluster_size[k]);
	}
	printf("Iterations:%d\n",iteracoes);
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

int main(int argc, char *argv[]){

	if (argc < 3) {
		printf("Not enough arguments!");
		return -1;
	}

	N = atoi(argv[1]);
	K = atoi(argv[2]);
	T = 1;

	if(argc >= 4) T = atoi(argv[3]);
	
	omp_set_num_threads(T);

	k_means_lloyd_algorithm();

	freeMemory();

	return 0;
}
