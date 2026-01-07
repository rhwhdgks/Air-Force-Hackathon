#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

void dfs(int x, int y, vector <vector<int>> &n, bool visited[], int &max_count, int count) {
    visited[n[x][y]] = true;
    max_count = max(max_count, count);
    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};

    for (int dir = 0; dir < 4; dir++) {
        int nx = x + dx[dir];
        int ny = y + dy[dir];
        if (nx >= 0 && nx < n.size() && ny >= 0 && ny < n[0].size() && !visited[n[nx][ny]]) {
            dfs(nx, ny, n, visited, max_count, count + 1);
        }
    }
    visited[n[x][y]] = false;
}
int main() {
   int r, c;
   cin >> r >> c;
   vector <vector<int>> n(r, vector<int>(c, 0));
   for (int i = 0; i < r; i++) {
        string s;
        cin >> s;
        for (int j = 0; j < c; j++) {
            n[i][j] = s[j] - 'A';
        }
   }
   bool visited[26] = {false};
   int max_count = 1;
   dfs(0, 0, n, visited, max_count, 1);
   cout << max_count;
   return 0;
}