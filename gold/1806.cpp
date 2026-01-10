#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
using namespace std;

int main() {
    int n, s;
    cin >> n >> s;
    vector <int> num(n, 0);
    vector <int> sum(n, 0);
    int answer = INT_MAX;
    for (int i = 0; i < n; i++) {
        cin >> num[i];
    }
    sum[0] = num[0];
    for (int i = 1; i < n; i++) {
        sum[i] = sum[i-1] + num[i];
    }
    if (sum[n-1] < s) {
        cout << 0;
        return 0;
    } 
    int left = 0;
        for (int right = 0; right < n; right++) {
            while (left <= right) {
                long long pop = sum[right] - sum[left] + num[left];
                if (pop >= s) {
                    answer = min(answer, right-left+1);
                    left++;
                }
                else break;
            }
        }
         cout << answer;  
}