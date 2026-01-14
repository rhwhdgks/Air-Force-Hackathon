#include <iostream>
#include <cmath>
using namespace std;

int answer(int x, int y) {  
    long long k = y-x;
    long long i = sqrt(k);
    long long m = i*i;
    long long n = i*(i+1);
    if (k == m) {
            return 2*i-1;
    }
    else if (k <= n) {
            return 2*i;
    }
    else {
            return 2*i+1;
    }
    
    
}
int main(){
    int t;
    cin >> t;

    for (int i = 0; i < t; i++) 
    {
        int x, y;
        cin >> x >> y;
        cout << answer(x, y) << endl; 
    }
}