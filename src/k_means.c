#include "../include/utils.h"

#define N 10000000
#define K 4

typedef struct point{
	float x;
	float y;
	int cluster_atribuido;
} point;

typedef struct cluster {
	point centroide;
	struct point points[N];
	int used; //basicamente para sabermos quantos elementos estão no cluster
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

		float menor_distancia = distancia_euclidiana(points[i],clusters[0].centroide);
		for (int j = 1;j < K; j++){
			float distancia = distancia_euclidiana(points[i],clusters[j].centroide);
			if(distancia < menor_distancia){
				menor_distancia = distancia;
				cluster_mais_proximo = j;
			}
		}
		points[i].cluster_atribuido = cluster_mais_proximo;
		remover_ponto_cluster(cluster_mais_proximo,i);
		adicionar_ponto_cluster(cluster_mais_proximo,points[i]);
	}
}

void remover_ponto_cluster(int k, int ind) { 
	//eu sei que isto é um bocado ineficiente mas para já fica assim só para testar e entretanto tentamos melhorar
	for(int i = ind; i < clusters[k].used; i++) clusters[k].points[i] = clusters[k].points[i + 1];
}

void adicionar_ponto_cluster(int k, point p) { //fiz esta função só pq ficava uma coisa enorme na de atribuir
	clusters[k].points[clusters[k].used++] = p;
}

point calcular_centroide(int k){
	int sum_x = 0;
	int sum_y = 0;
	point centroide;

	for(int i=0; i<N; i++) {
		sum_x += clusters[k].centroide.x;
		sum_y += clusters[k].centroide.y;
	}
	centroide.x = sum_x/N;
	centroide.y = sum_y/N;

	return centroide;
}

//Cria os pontos aleatórios, clusters e atribui o cluster mais próximo a cada ponto
void inicializa() {

	srand(10);
	for(int i = 0; i < N; i++) {
		points[i].x = (float) rand() / RAND_MAX;
		points[i].y = (float) rand() / RAND_MAX;;
		}

	for(int i = 0; i < K; i++) {
		clusters[i].centroide.x =  points[i].x;
		clusters[i].centroide.y = points[i].y;
		clusters[i].used = 0;
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