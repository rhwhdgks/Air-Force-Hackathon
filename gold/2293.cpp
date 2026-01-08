#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int answer(vector<int> &coin, int k) {
    vector<long long> dp(k+1, 0);
    dp[0] = 1;

    for (int i = 0; i < coin.size(); i++) {
        for (int j = coin[i]; j <= k; j++) {
            dp[j] += dp[j - coin[i]];
        }
    }
    return coin[k];
}
int main() {
    int n, k;
    vector <int> coin;
    cin >> n >> k;
    vector <int> dp(k+1, 0);
    for (int i = 0; i < n; i++) {
        int price;
        cin >> price;
        coin.push_back(price);
    }
    sort(coin.begin(), coin.end());
    dp[0] = 1;

    cout << answer(coin, k);
}