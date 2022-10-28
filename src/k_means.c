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
	int used; //número de elementos no cluster
} cluster;


struct point points[N];
struct cluster clusters[K];
struct point centroides_antigos[K];

int iteracoes;


//Função que calcula o centroide de um cluster
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

//Função que calcula a distância euclidiana entre dois pontos
float distancia_euclidiana(point a, point b){
	return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}


//Compara os centroides antigos com os novos, se forem diferentes devolve 0 senão devolve 1
int comparar_centroides(){
	for(int i = 0; i < K; i++){
		if(clusters[i].centroide.x != centroides_antigos[i].x || clusters[i].centroide.y != centroides_antigos[i].y) return 0;
	}
	return 1;
}

//Função que guarda os clusters antigos e reinicia a contagem de elementos
void reset_clusters(){
	for (int i = 0; i < K; i++){
		centroides_antigos[i] = clusters[i].centroide; 
		clusters[i].used = 0;
	}
}

//Função que adiciona um ponto a um determinado cluster
void adicionar_ponto_cluster(int k, point p) {
	clusters[k].points[clusters[k].used++] = p;
}

//Função que atruibui cada ponto ao seu cluster mais próximo
//Se reatribuir os pontos então devolve 0, senão devolve 1
int atribuir_clusters() {
	int cluster_mais_proximo;
	float menor_distancia, distancia;
	point cent = clusters[0].centroide, p;

	reset_clusters();

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
			/*Tentativa de tornar o código mais vetorizável
			cluster_mais_proximo = (distancia < menor_distancia) ? j : cluster_mais_proximo;
			menor_distancia = (distancia < menor_distancia) ? distancia : menor_distancia;
			*/

		}
		adicionar_ponto_cluster(cluster_mais_proximo,p);
	}
	for (int k = 0; k < K; k++){
		clusters[k].centroide = calcular_centroide(k);
	}
	return (iteracoes == 0) ? 0 : comparar_centroides();
}


//Função que preenche um array com pontos aleatórios e inicia os clusters
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


//Função principal que aplica o algoritmo de Lloyd
void k_means_lloyd_algorithm() {
	iteracoes = 0;

	inicializa();

	while(!atribuir_clusters()) {
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