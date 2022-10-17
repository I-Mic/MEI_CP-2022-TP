#include "../include/utils.h"

#define N 10000000
#define K 4

typedef struct point{
	float x;
	float y;
} point;

typedef struct cluster {
	point centroide;
	struct point points[N];
	int used; //basicamente para sabermos quantos elementos estão no cluster
} cluster;

struct point points[N];
struct cluster clusters[K];

struct cluster last_clusters[K]; //variável que irá guardar o último estado dos clusters


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

//Função que atruibui cada ponto ao seu cluster mais próximo
void atribuir_cluster_inicial(){
	for (int i=0; i < N; i++){
		int cluster_mais_proximo = 0;
		//possivel melhoria de performance aqui, usando uma variavel point em vez
		//de ir sempre ao array buscar o ponto.

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
//Se reatribuir os pontos entao devolve 1, senao devolve 0
int reatribuir_clusters() {
	int has_changed = 0,test =0;
	//guardar_estado_clusters(); //guarda o estado dos clusters antes de estes serem alterados

	for (int i=0; i<K; i++) { //para cada cluster

		for (int j=0; j<clusters[i].used; j++) { //para cada ponto do cluster
			int point_changed = 0;
			int cluster_mais_proximo = j;
			float menor_distancia = distancia_euclidiana(clusters[i].points[j],clusters[i].centroide);
			//printf("%d ",j);
			for (int k = 0; k < K; k++){
				//para evitar fazer isto caso seja o mesmo cluster
				if(k != i){
					float distancia = distancia_euclidiana(clusters[i].points[j],clusters[k].centroide);
					
					if(distancia < menor_distancia){
						menor_distancia = distancia;
						cluster_mais_proximo = k;
						point_changed ++;
					}
				}

			}
			if(point_changed != 0){
				adicionar_ponto_cluster(cluster_mais_proximo,clusters[i].points[j]);
				remover_ponto_cluster(i,j); //i-cluster, j-posição no array de pontos do cluster
				has_changed = 1;
				//printf("I did smth %d\n",j);
				j--;
			}
		}
		
	}
	if(has_changed != 0){
		for (int i=0; i<K;i++){
			clusters[i].centroide = calcular_centroide(i);
		}
	}
	//Assim sabemos sempre se esta funcao fez alguma alteracao aos clusters
	return has_changed;
}

//Função que remove um ponto de um determinado cluster
void remover_ponto_cluster(int k, int ind) { 
	clusters[k].points[ind] = clusters[k].points[clusters[k].used-1];
	clusters[k].used--;
}

//Função que adiciona um ponto a um determinado cluster
void adicionar_ponto_cluster(int k, point p) { //fiz esta função só pq ficava uma coisa enorme na de atribuir
	clusters[k].points[clusters[k].used++] = p;
}

/*
//Função que guarda o estado de um cluster
void guardar_estado_clusters() {
	for(int i = 0; i < K; i++) {
		last_clusters[i].centroide = clusters[i].centroide;
		last_clusters[i].used = clusters[i].used; 
		for(int j = 0; j < clusters[i].used; j++) {
			last_clusters[i].points[j] = clusters[i].points[j];
		}
	}
}
*/


//Cria os pontos aleatórios, clusters e atribui o cluster mais próximo a cada ponto
void inicializa() {

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
}
/*
//Função que verifica se o algoritmo já convergiu
int has_converged() {
	//basicamente a ideia será guardar o estado dos clusters antes de passar para a próxima iteração
	//depois comparar os pontos de todos
	//se não tiver havido alteração de uma iteração para outra então o algoritmo acabou/convergiu

	for(int i = 0; i < K; i++) {
		if (last_clusters[i].centroide.x != clusters[i].centroide.x
		|| last_clusters[i].centroide.y != clusters[i].centroide.y 
		|| last_clusters[i].used != clusters[i].used) return 0;

		for(int j = 0; j < clusters[i].used; j++) {
			if(last_clusters[i].points[j].x != clusters[i].points[j].x 
			|| last_clusters[i].points[j].y != clusters[i].points[j].y) return 0;
		}
	}
	
	return 1;
}
*/
//Função principal que aplica o algoritmo de Lloyd
void k_means_lloyd_algorithm() {
	int iteracoes = 1;

	inicializa();
	printf("N = %d, K = %d\n",N,K);

	while(reatribuir_clusters()) {

		iteracoes++;
	}
	for(int k=0; k < K;k++){
		printf("Center: (%0.3f, %0.3f) : Size: %d\n",clusters[k].centroide.x,clusters[k].centroide.y,clusters[k].used);
	}
	printf("Iterations:%d\n",iteracoes);
}

void main(){

	clock_t begin = clock();

	k_means_lloyd_algorithm();


	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Exec time: %f secs\n",time_spent);
	return;
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