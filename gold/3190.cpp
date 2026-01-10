#include <iostream>
#include <vector>
#include <queue>
#include <deque>
using namespace std;

int main() {
    //dummy라고 하는 게임판을 벡터 2차원으로 구현하고 사과는 1, 아무것도 없는 곳은 0, 뱀은 2로 설정한다. 그 이후 사과와 회전 정보를 받는다.
    int n,k,l;
    cin >> n >> k;
    vector<vector<int>> dummy(n, vector<int>(n,0));
    queue<pair<int, char>> q;
    for (int i = 0; i < k; i++) {
        int r, c;
        cin >> r >> c;
        dummy[r-1][c-1] = 1;
    }
    cin >> l;
    for (int i = 0; i < l; i++) {
        int time;
        char c;
        cin >> time >> c;
        q.push({time, c});
    }
    //row와 coloum을 변수로 두고 그 움직임을 head로 두고, 뱀의 머리부터 다시 움직여야하기에 deque를 활용하여 뱀의 위치를 구현한다.
    int r = 0, c = 0, head = 0;
    int answer = 0;

    int rmove[] = {0, 1, 0, -1};
    int cmove[] = {1, 0, -1, 0};

    deque<pair<int, int>> snake;
    dummy[0][0] = 2;
    snake.push_back({0,0});
    //게임이 끝날때까지 반복되는 반복문을 바탕으로 사과와 관련된 부분과 회전에 관련된 부분, 게임이 끝나는 경우에 대한 부분 등 3가지 주요 과정으로 나누어 구현
    while (1){  
        answer++;
        int nr = r + rmove[head];
        int nc = c + cmove[head];
        
       if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
        break;
       }

       snake.push_back({nr, nc});

       if (dummy[nr][nc] == 1) {
        dummy[nr][nc] = 2;
       }
       else if (dummy[nr][nc] == 0) {
        dummy[nr][nc] = 2;
        dummy[snake.front().first][snake.front().second] = 0;
        snake.pop_front();
       }
       else break;

       r = nr;
       c = nc;

        if (!q.empty() && q.front().first == answer) {
            if(q.front().second == 'L') {
                if (head == 0) {
                    head = 3;
                }
                else head -= 1;
            }
            else {
                if (head == 3) {
                    head = 0;
                }
                else head += 1;
            }
            q.pop();
        }
    }
    cout << answer;

}