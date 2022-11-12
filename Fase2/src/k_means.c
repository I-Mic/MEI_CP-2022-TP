#include "../include/utils.h"

typedef struct point{
	float x;
	float y;
} point;

typedef struct cluster {
	point centroide;
	struct point *points;
	int used; //number of elements in the cluster
} cluster;

//Struct of global cluster doesnt require to have the array of points
typedef struct global_cluster {
	point centroide;
	int used; //number of elements in the cluster
} global_cluster;

int N;								//Nr of total Points generated
int K;								//Nr of total clusters
int T;								//Nr of total threads

struct point* points;				//Array of generated points
struct global_cluster *clusters;	//The global array of clusters
struct cluster *thread_clusters;	//The local-thread array of clusters
struct point *centroides_antigos;	//Previous centroids 


//Calculates the centroide of a cluster
void calcular_centroide(){

	#pragma omp parallel for
	for (int k = 0; k < K; k++)
	{
		float sum_x = 0;
		float sum_y = 0;
		int total = 0;

		for (int t = 0; t < T;t++){
			int pos = t * K + k;
			total += thread_clusters[pos].used;

			for (int i = 0; i < thread_clusters[pos].used; i++){
				sum_x += thread_clusters[pos].points[i].x;
				sum_y += thread_clusters[pos].points[i].y;
			}
		}
		//ads centroid and total points nr to global cluster
		clusters[k].used = total;
		clusters[k].centroide.x = sum_x/total;
		clusters[k].centroide.y = sum_y/total;
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
		if(clusters[i].centroide.x != centroides_antigos[i].x || clusters[i].centroide.y != centroides_antigos[i].y) return 0;
	}
	return 1;
}

//Resets the clusters, saving the current centroids, for the next iteration
//Added paralelism for better performance
void reset_clusters(){
	#pragma omp parallel for
	for (int k = 0; k < K; k++){
		centroides_antigos[k] = clusters[k].centroide; 
		for (int t = 0; t < T; t++)
		{
			thread_clusters[t * K + k].used = 0;
		}
	}
}

//Function that adds a point to the cluster of certain thread
void adicionar_ponto_cluster_thread(int t,int k, point p) {
	thread_clusters[t * K + k].points[thread_clusters[t * K + k].used++] = p;
}


//Function that designates a point to the closest cluster with paralelism
//If any cluster changed at the end returns 1, otherwise returns 0.
int atribuir_clusters() {

	#pragma omp parallel for 
	for (int i = 0; i < N; i++){
		int cluster_mais_proximo = 0;
		point cent = clusters[0].centroide, p = points[i];

		float menor_distancia = distancia_euclidiana(p,cent);

		for (int j = 1; j < K; j++){
			float distancia = distancia_euclidiana(p,clusters[j].centroide);

				if(distancia < menor_distancia){
					cluster_mais_proximo = j;
					menor_distancia = distancia;
				}		
		}

		//Adds the point to the correct thread_cluster
		int threadId = omp_get_thread_num();
		adicionar_ponto_cluster_thread(threadId,cluster_mais_proximo,p);
		
	}

	calcular_centroide();

	return comparar_centroides();
}

//Creates N Random points and assigns the first K points as centroids of each cluster
void inicializa() {
	points = (point*) malloc(N * sizeof(point));
	clusters = (global_cluster*) malloc(K * sizeof(global_cluster));
	centroides_antigos = (point*) malloc(K * sizeof(point));
	thread_clusters = (cluster*) malloc((K * T) * sizeof(cluster));
	
	srand(10);
	for(int i = 0; i < N; i++) {
		points[i].x = (float) rand() / RAND_MAX;
		points[i].y = (float) rand() / RAND_MAX;
		}

	#pragma omp parallel for 
	for(int i = 0; i < K; i++) {
		clusters[i].centroide.x = points[i].x;
		clusters[i].centroide.y = points[i].y;
		clusters[i].used = 0;

		for (int j = 0; j < T;j++){
			thread_clusters[j * K + i].used = 0;
			thread_clusters[j * K + i].points = (point*) malloc(((N / T) + 1) * sizeof(point));
		}	 
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
		printf("Center: (%0.3f, %0.3f) : Size: %d\n",clusters[k].centroide.x,clusters[k].centroide.y,clusters[k].used);
	}
	printf("Iterations:%d\n",iteracoes);
}

//Clears memory allocated for the arrays
void freeMemory(){
		free(thread_clusters);
		free(points);
		free(clusters);
		free(centroides_antigos);
}

int main(int argc, char *argv[]){
	N = atoi(argv[1]);
	K = atoi(argv[2]);
	T = 1;
	if(argc >= 4){
		T = atoi(argv[3]);
	}

	//The best result will mostly be nr threads = 2 * nr clusters
	omp_set_num_threads(T);

	k_means_lloyd_algorithm();

	freeMemory();

	return 0;
}
