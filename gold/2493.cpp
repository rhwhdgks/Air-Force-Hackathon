#include <iostream>
#include <vector>
#include <stack>
using namespace std;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    stack <pair<int, int>> st;
    vector <int> answer(n+1, 0);
    for (int i = 1; i <= n; i++) {
        int k;
        cin >> k;

        while (!st.empty() && st.top().first < k) {
            st.pop();
        }
       
        answer[i] = st.empty() ? 0 : st.top().second;
        st.push({k, i});
    }
    for (int i = 1; i <= n; i++) {
        cout << answer[i] << " ";
    }
}