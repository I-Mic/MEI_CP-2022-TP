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
	int used; //basicamente para sabermos quantos elementos estão no cluster
} cluster;


struct point *points;
struct cluster clusters[K];
struct point centroides_antigos[K];



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
	return sqrt(((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)));
}


//Compara os centroides antigos com os novos, se forem diferentes devolve 0 senao devolve 1
int comparar_centroides(){
	for(int i=0;i<K;i++){
		if(clusters[i].centroide.x != centroides_antigos[i].x || clusters[i].centroide.y != centroides_antigos[i].y) return 0;
	}
	return 1;
}

void reset_clusters(){
	for (int i=0; i<K;i++){
		centroides_antigos[i] = clusters[i].centroide; 
		clusters[i].used = 0;
	}
}

//Função que adiciona um ponto a um determinado cluster
void adicionar_ponto_cluster(int k, point p) { //fiz esta função só pq ficava uma coisa enorme na de atribuir
	clusters[k].points[clusters[k].used] = p;
	clusters[k].used++;
}


//Função que atruibui cada ponto ao seu cluster mais próximo
void atribuir_cluster_inicial(){
	for (int i=0; i < N; i++){
		int cluster_mais_proximo = 0;
		float menor_distancia = distancia_euclidiana(points[i],clusters[0].centroide);

		for (int j = 1; j < K; j++){
			float distancia = distancia_euclidiana(points[i],clusters[j].centroide);

			if(distancia < menor_distancia){
				menor_distancia = distancia;
				cluster_mais_proximo = j;
			}
		}
		adicionar_ponto_cluster(cluster_mais_proximo,points[i]);
	}
	for (int k=0;k<K;k++){
		clusters[k].centroide = calcular_centroide(k);
	}
}

//Provavelmente dá para juntar esta função com a de cima de alguma maneira pq são bastante parecidas
//Função que percorre todos os pontos de todos os clusters e lhes atribiu (ou não) um novo cluster
//Se reatribuir os pontos entao devolve 0, senao devolve 1
int reatribuir_clusters() {
	reset_clusters();
	int has_changed = 0;

	for (int i = 0;i<N;i++){
		int cluster_mais_proximo = 0;

		float menor_distancia = distancia_euclidiana(points[i],clusters[0].centroide);

		for (int j = 1; j < K; j++){
			
			float distancia = distancia_euclidiana(points[i],clusters[j].centroide);

			if(distancia < menor_distancia){
				menor_distancia = distancia;
				cluster_mais_proximo = j;
			}
		}
		adicionar_ponto_cluster(cluster_mais_proximo,points[i]);
	}
	for (int k=0;k<K;k++){
		clusters[k].centroide = calcular_centroide(k);
	}
	return comparar_centroides();
}


//Cria os pontos aleatórios, clusters e atribui o cluster mais próximo a cada ponto
void inicializa() {
	points = (struct point *)malloc(N*sizeof(struct point));
	for(int i= 0;i<K;i++) clusters[i].points = (struct point *)malloc(N*sizeof(struct point));

	srand(10);
	for(int i = 0; i < N; i++) {
		points[i].x = (float) rand() / RAND_MAX;
		points[i].y = (float) rand() / RAND_MAX;
		}

	for(int i = 0; i < K; i++) {
		clusters[i].centroide.x =  points[i].x;
		clusters[i].centroide.y = points[i].y;
		clusters[i].used = 0;
	}

	atribuir_cluster_inicial();
	printf("\n");
}


//Função principal que aplica o algoritmo de Lloyd
void k_means_lloyd_algorithm() {
	int iteracoes = 1;

	inicializa();
	printf("N = %d, K = %d\n",N,K);

	while(!reatribuir_clusters()) {
		iteracoes++;
	}

	for(int k=0; k < K;k++){
		printf("Center: (%0.3f, %0.3f) : Size: %d\n",clusters[k].centroide.x,clusters[k].centroide.y,clusters[k].used);
	}
	printf("Iterations:%d\n",iteracoes);
}

int main(){
	clock_t begin = clock();

	k_means_lloyd_algorithm();


	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Exec time: %f secs\n",time_spent);
	return 0;

}