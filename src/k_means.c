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

float distancia(point a, point b){
	return sqrt(((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)));
}

void atribuir_cluster(){

}


void main(){
	inicializa();
	for(int i = 0; i < N; i++) {
		printf("(%f,%f)",points[i].x,points[i].y);
	}
}