#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
using namespace std;
struct Fish {
    int size;
    int x;
    int y;
};
struct dinner {
    int d;
    int x;
    int y;
};
bool cmp(const dinner& a, const dinner& b) {
    if (a.d != b.d) return a.d < b.d;
    if (a.x != b.x) return a.x < b.x;
    return a.y < b.y;
}
int main(){
    int n; 
    cin >> n;
    int space[20][20];
    vector <Fish> f;
    int sharkx, sharky, sharksize;
    sharksize = 2;

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) {
            cin >> space[i][j];
            if(space[i][j] == 9) {
                    sharkx = i;
                    sharky = j;
                }
            }
    }
    int answer = 0;
    while (true) {
        int d = 100;
        int visited[20][20] = {0,};
        int dist[20][20] = {0,};
        dist[sharkx][sharky] = 0;
        vector <dinner> eat; 
        queue <pair<int, int>> move;
        move.push({sharkx, sharky});
        while(!move.empty()){
            auto [x, y] = move.front(); move.pop();
            visited[x][y] = 1;
            int dix[] = {0, -1, 1, 0};
            int diy[] = {1, 0, 0, -1};
            for (int i = 0; i < 4; i++) {
                int x1, y1;
                x1 = x + dix[i];
                y1 = y + diy[i];
                if (x1 < n && x1 >= 0 && y1 < n && y1 >= 0 && !visited[x1][y1]) {
                    if (space[x1][y1] == 0 || space[x1][y1] == sharksize) {
                        move.push({x1, y1});
                        dist[x1][y1] = dist[x][y] + 1;
                    }
                    else if (space[x1][y1] < sharksize && space[x1][y1] != 0) {
                        dist[x1][y1] = dist[x][y]+1;
                        eat.push_back({dist[x1][y1], x1, y1});
                        move.push({x1, y1});
                    }
                    else continue;
                }
            }

        }
        if (!eat.empty()) break;
        else {
            answer += eat[0].d;

            space[sharkx][sharky] = 0;
            sharkx = eat[0].x;
            sharky = eat[0].y;
            space[sharkx][sharky] = 9;
        }

    }

    

}