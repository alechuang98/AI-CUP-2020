#include <bits/stdc++.h>
using namespace std;
#define IOS ios_base::sync_with_stdio(0); cin.tie(0)

const int MXN = 2005, B = 128, INF = 1e9 + 7;
const char *data_path = "AIcup_testset_ok";
const char *tmp_path = "tmp";

int n, m = 1001;
vector<int> pitch;
vector<vector<int>> med, sum, p_med, p_sum;
vector<vector<int>> dp, first_pitch;

void init(){
	med = vector<vector<int>> (n, vector<int> (B, 0));
	sum = vector<vector<int>> (n, vector<int> (B, 0));
	p_sum = vector<vector<int>> (n, vector<int> (B, 0));
	p_med = vector<vector<int>> (n, vector<int> (B, 0));
	for(int i = 1; i < n; i ++){
		for(int j = 0; j < B; j ++){
			med[i][j] = med[i - 1][j];
			sum[i][j] = sum[i - 1][j];
		}
		med[i][pitch[i]] ++;
		sum[i][pitch[i]] += pitch[i];
		p_med[i][0] = med[i][0];
		p_sum[i][0] = sum[i][0];
		for(int j = 1; j < B; j ++){
			p_med[i][j] = p_med[i][j - 1] + med[i][j];
			p_sum[i][j] = p_sum[i][j - 1] + sum[i][j];
		}
	}
}
int get_median(int l, int r){
	int L = -1, R = B, num = (r - l + 1) >> 1;
	while(L < R - 1){
		int mid = (L + R) >> 1;
		if(p_med[r][mid] - p_med[l - 1][mid] <= num) L = mid;
		else R = mid;
	}
	return R;
}

int loss(int l, int r){
	if(l < 1) return INF;
	int median = get_median(l, r), res = 0;
	if(median > 0) res += (p_med[r][median - 1] - p_med[l - 1][median - 1]) * median - (p_sum[r][median - 1] - p_sum[l - 1][median - 1]);
	if(median < B){
		res += 
		((p_sum[r][B - 1] - p_sum[r][median]) - (p_sum[l - 1][B - 1] - p_sum[l - 1][median])) -
		((p_med[r][B - 1] - p_med[r][median]) - (p_med[l - 1][B - 1] - p_med[l - 1][median])) * median;
	}
	return res;
}
int sol(int error_bound){
	dp = vector<vector<int>> (m, vector<int> (n, INF));
	first_pitch = vector<vector<int>> (m, vector<int> (n, 1));
	dp[0][0] = first_pitch[0][0] = 0;
	for(int i = 1; i < m; i ++){
		for(int j = 1; j < n; j ++){
			int a = dp[i - 1][first_pitch[i][j - 1] - 1] + loss(first_pitch[i][j - 1], j), b = dp[i - 1][j - 1];
			dp[i][j] = min(a, b);
			if(a < b) first_pitch[i][j] = first_pitch[i][j - 1];
			else first_pitch[i][j] = j;
		}
		if(dp[i][n - 1] < error_bound) return i;
	}
	return m - 1;
}
			
int main(int argc, char *argv[]){
	clock_t start = clock();
	int id = atoi(argv[1]);
	int error_bound = atoi(argv[2]);
	printf("start %4d-th song.\n", id);
	char file[105];
	float tmp;
	sprintf(file, "%s/%d/%d_pitch.txt", data_path, id, id);
	
	FILE *fp = fopen(file, "r");
	pitch.push_back(0);
	while(fscanf(fp, "%f", &tmp) != EOF){
		pitch.push_back((int)(tmp + 0.5));
	}
	fclose(fp);
	n = pitch.size();
	init();

	int note_num = sol(error_bound), cur_pitch = n - 1;
	
	vector<vector<int>> ans;
	for(int i = note_num; i > 0; i --){
		int median = get_median(first_pitch[i][cur_pitch], cur_pitch);
		if(median > 0){
			ans.push_back(vector<int> {first_pitch[i][cur_pitch], cur_pitch, median});
		}
		cur_pitch = first_pitch[i][cur_pitch] - 1;
	}
	reverse(ans.begin(), ans.end());
	
	sprintf(file, "%s/%d.txt", tmp_path, id);
	fp = fopen(file, "w");
	for(auto i : ans){
		fprintf(fp, "%.4f %.4f %d\n", i[0] * 0.032 + 0.016, i[1] * 0.032 + 0.016, i[2]);
	}
	fclose(fp);
	printf("number of music notes in %4d-th song: %4d, using %.2f s\n", id, (int)ans.size(), (float)(clock() - start) / CLOCKS_PER_SEC);
	return 0;
}

