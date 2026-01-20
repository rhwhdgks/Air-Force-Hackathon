#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;
int dist[1001][1001] = {0,};
void dijkstra(int i, vector<vector<pair<int, int>>>& v) {

    priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> pq;
    
    pq.push({0, i});
    dist[i][i] = 0;
    while(!pq.empty()) {
        auto [c, s] = pq.top(); pq.pop();

        if (dist[i][s] < c) continue;

        for (auto [j, k] : v[s]) {
            if (k+c < dist[i][j]) {
                dist[i][j] = k+c;
                pq.push({dist[i][j], j});
            }
        }
    }
    return;
}
int main() {
    int n, m, x;
    cin >> n >> m >> x;
    vector<vector<pair<int,int>>> v(1001);
    
    for (int i = 1; i <= m; i++) {
        int a, b, c;
        cin >> a >> b >>c;
        v[a].push_back({b, c});
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            dist[i][j] = 1e9;
        }
    }
    int ans = 0;
    for (int i = 1; i <= n; i++) {
        dijkstra(i, v);
    }
    for (int i = 1; i <= n; i++) {
        if (dist[i][x] != 1e9 && dist[x][i] != 1e9) {
            ans = max(ans, dist[i][x] + dist[x][i]);
        }
    }
    cout << ans;
}