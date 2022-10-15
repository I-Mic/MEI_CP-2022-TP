#include "../include/utils.h"


#define N 10000000
#define K 4

typedef struct point{
	float x;
	float y;
	int cluster_atribuido;
}point;

struct point points[N];
struct point cluster[K];

//Cria os pontos aleatorios e os clusters
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
		cluster[i].x = x;
		cluster[i].y = y;
	}
}

//Funcao que calcula a distancia euclidiana entre dois pontos
float distancia_euclidiana(point a, point b){
	return sqrt(((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)));
}


//Funcao que atribui aos pontos o seu cluster mais proximo
void atribuir_cluster(){
	for (int i=0; i < N; i++){
		int cluster_mais_proximo = 0;

		//possivel melhoria de performance aqui, usando uma variavel point enves
		//de ir sempre ao array buscar o ponto.
		
		float menor_distancia = distancia_euclidiana(points[i],cluster[0]);
		for (int j = 1;j < K; j++){
			float distancia = distancia_euclidiana(points[i],cluster[j]);
			if(distancia < menor_distancia){
				menor_distancia = distancia;
				cluster_mais_proximo = j;
			}
		}
		points[i].cluster_atribuido = cluster_mais_proximo;
	}
}


void main(){
	inicializa();
	atribuir_cluster();
	for (int i=0;i<K;i++){
		printf("Cluester = (%f , %f)\n",cluster[i].x,cluster[i].y);
	}
	for(int i = 0; i < 100; i++) {
		printf("Ponto (%f , %f)-Cluster atribuido = %d\n",points[i].x,points[i].y,points[i].cluster_atribuido);
	}
}