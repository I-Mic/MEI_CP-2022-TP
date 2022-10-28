#include "../include/utils.h"

#define N 10000000
#define K 4

typedef struct point{
	float x;
	float y;
} point;

typedef struct cluster {
	point centroide;
	struct point *points;
	int used; //number of elements in the cluster
} cluster;


struct point points[N];
struct cluster clusters[K];
struct point centroides_antigos[K];


//Calculates the centroide of a cluster
point calcular_centroide(int k){
	float sum_x = 0;
	float sum_y = 0;
	point centroide;

	for(int i=0; i<clusters[k].used; i++) {
		sum_x += clusters[k].points[i].x;
		sum_y += clusters[k].points[i].y;
	}
	centroide.x = sum_x/clusters[k].used;
	centroide.y = sum_y/clusters[k].used;

	return centroide;
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
void reset_clusters(){
	for (int i = 0; i < K; i++){
		centroides_antigos[i] = clusters[i].centroide; 
		clusters[i].used = 0;
	}
}

//Function that adds a point to a cluster
void adicionar_ponto_cluster(int k, point p) {
	clusters[k].points[clusters[k].used++] = p;
}

//Function that designates a point to the closest cluster
//If any cluster changed at the end returns 1, otherwise returns 0.
int atribuir_clusters() {
	int cluster_mais_proximo;
	float menor_distancia, distancia;
	point cent = clusters[0].centroide, p;

	for (int i = 0; i < N; i++){
		p = points[i];
		cluster_mais_proximo = 0;

		menor_distancia = distancia_euclidiana(p,cent);

		for (int j = 1; j < K; j++){
			
			distancia = distancia_euclidiana(p,clusters[j].centroide);

			if(distancia < menor_distancia){
				cluster_mais_proximo = j;
				menor_distancia = distancia;
			}
			/*Attempt at vectorizing the code
			cluster_mais_proximo = (distancia < menor_distancia) ? j : cluster_mais_proximo;
			menor_distancia = (distancia < menor_distancia) ? distancia : menor_distancia;
			*/

		}
		adicionar_ponto_cluster(cluster_mais_proximo,p);
	}
	for (int k = 0; k < K; k++){
		clusters[k].centroide = calcular_centroide(k);
	}
	return comparar_centroides();
}

//Creates N Random points and assigns the first K points as centroids of each cluster
void inicializa() {

	srand(10);
	for(int i = 0; i < N; i++) {
		points[i].x = (float) rand() / RAND_MAX;
		points[i].y = (float) rand() / RAND_MAX;
		}

	for(int i = 0; i < K; i++) {
		clusters[i].centroide.x = points[i].x;
		clusters[i].centroide.y = points[i].y;
		clusters[i].used = 0;
		clusters[i].points = (struct point*) malloc(N * sizeof(struct point));
	}
}


//Function that aplies Lloyd algorithm
//Only ends when the aplication has converged 
void k_means_lloyd_algorithm() {
	int iteracoes = 0;

	inicializa();

	while(!atribuir_clusters()) {
		reset_clusters();
		iteracoes++;
	}

	printf("N = %d, K = %d\n",N,K);

	for(int k = 0; k < K; k++){
		printf("Center: (%0.3f, %0.3f) : Size: %d\n",clusters[k].centroide.x,clusters[k].centroide.y,clusters[k].used);
	}
	printf("Iterations:%d\n",iteracoes);
}

int main(){
	k_means_lloyd_algorithm();
	return 0;
}