#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int main() {
    int n;
    cin >> n;
    vector<int> a(n, 0);

    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }
    vector <int> LIS(n, 1);
    
    
    //한 지점을 기준으로 점차 증가하는 수열을 찾는 과정
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (a[j] < a[i]) {
                LIS[i] = max(LIS[i], LIS[j]+1);
            }
        }
    }
    
    //한 지점을 기준으로 점차 감소하는 수열을 찾는 과정
    vector <int> LDS(n, 1);
    for (int i = n-1; i >= 0; i--) {
        for (int j = n-1; j > i; j--) {
            if (a[i] > a[j]) {
                LDS[i] = max(LDS[i], LDS[j]+1);
            }
        }
    }
    int ans = 0;
    for (int i = 0; i < n; i++) {
        ans = max(ans, LIS[i]+LDS[i]-1);
    }

    cout << ans;

}