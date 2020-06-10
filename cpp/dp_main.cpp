#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
using namespace std;

int main(int argc, char *argv[]){
	int id = 1, cnt = 0, cur = 0;
	int error_bound = atoi(argv[1]);
	int max_core = atoi(argv[2]);
	char id_s[105], eb_s[105];
	sprintf(eb_s, "%d", error_bound);
	while(cnt < 1500){
		while(cur < max_core && id <= 1500){
			int pid = fork();
			if(pid == 0){
				sprintf(id_s, "%d", id);
				execlp("./cpp/dp", "./cpp/dp", id_s, eb_s);
			}
			id ++;
			cur ++;
		}
		wait(NULL);
		cur --;
		cnt ++;
	}
}

