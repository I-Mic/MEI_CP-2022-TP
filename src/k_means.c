#include "../include/utils.h"

#define N 10000000
#define K 4

typedef struct point{
	float x;
	float y;
	int cluster_atribuido;
} point;

typedef struct cluster {
	point p;
	struct point points[N];
} cluster;

struct point points[N];
struct cluster clusters[K];


//Função que calcula a distância euclidiana entre dois pontos
float distancia_euclidiana(point a, point b){
	return sqrt(((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)));
}

//Função que atribui aos pontos o seu cluster mais próximo
void atribuir_cluster(){
	for (int i=0; i < N; i++){
		int cluster_mais_proximo = 0;

		//possivel melhoria de performance aqui, usando uma variavel point enves
		//de ir sempre ao array buscar o ponto.

		float menor_distancia = distancia_euclidiana(points[i],clusters[0].p);
		for (int j = 1;j < K; j++){
			float distancia = distancia_euclidiana(points[i],clusters[j].p);
			if(distancia < menor_distancia){
				menor_distancia = distancia;
				cluster_mais_proximo = j;
			}
		}
		points[i].cluster_atribuido = cluster_mais_proximo;
	}
}

point calcular_centroide(int k){
	int sum_x = 0;
	int sum_y = 0;
	point centroide;

	for(int i=0; i<N; i++) {
		sum_x += clusters[k].p.x;
		sum_y += clusters[k].p.y;
	}
	centroide.x = sum_x/N;
	centroide.y = sum_y/N;

	return centroide;
}

//Cria os pontos aleatórios, clusters e atribui o cluster mais próximo a cada ponto
void inicializa() {

	srand(10);
	for(int i = 0; i < N; i++) {
		float x = (float) rand() / RAND_MAX;
		float y = (float) rand() / RAND_MAX;
		points[i].x = x;
		points[i].y = y;
		}

	for(int i = 0; i < K; i++) {
		float x = points[i].x;
		float y = points[i].y;
		clusters[i].p.x = x;
		clusters[i].p.y = y;
	}
	atribuir_cluster();
}







void main(){

	inicializa();

	/*
	//Debugging parece estar a atribuir os clusters corretamente
	for (int i=0;i<K;i++){
		printf("Cluester = (%f , %f)\n",cluster[i].x,cluster[i].y);
	}
	for(int i = 0; i < 100; i++) {
		printf("Ponto (%f , %f)-Cluster atribuido = %d\n",points[i].x,points[i].y,points[i].cluster_atribuido);
	}
	*/

}