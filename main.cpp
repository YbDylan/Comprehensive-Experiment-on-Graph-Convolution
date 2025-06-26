#include <iostream>
#include <vector>
#include <cmath>
#include <queue>

using namespace std;

// 输出邻接矩阵
void printMatrix(const vector<vector<int>>& A, const string& title) {
    cout << title << endl;
    for (const auto& row : A) {
        for (int val : row) cout << val << " ";
        cout << endl;
    }
}

// 输出归一化邻接矩阵
void printNormalizedMatrix(const vector<vector<double>>& A, const string& title) {
    cout << title << endl;
    for (const auto& row : A) {
        for (double val : row) printf("%.2f ", val);
        cout << endl;
    }
}

// 读取图并添加自环
void readGraph(int n, int m, vector<vector<int>>& A) {
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        A[u][v] = 1;
        A[v][u] = 1; // 无向图
    }
    for (int i = 0; i < n; ++i) {
        A[i][i] = 1;
    }
}

// 判断是否连通
bool isConnected(int n, const vector<vector<int>>& A) {
    vector<bool> visited(n, false);
    queue<int> q;
    q.push(0);
    visited[0] = true;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v = 0; v < n; ++v) {
            if (A[u][v] && !visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    for (bool v : visited) if (!v) return false;
    return true;
}

// 构建 CSR 格式
void buildCSR(int n, const vector<vector<int>>& A, vector<int>& rowPtr, vector<int>& colInd, vector<double>& values) {
    rowPtr.assign(n+1, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A[i][j]) {
                colInd.push_back(j);
                values.push_back(1.0);
            }
        }
        rowPtr[i+1] = colInd.size();
    }
}

// GCN 前向传播
vector<vector<double>> gcnForward(int n, int inF, int outF, const vector<int>& rowPtr, const vector<int>& colInd, const vector<double>& values, vector<vector<double>>& AH, vector<vector<double>>& W_out) {
    vector<double> deg(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int idx = rowPtr[i]; idx < rowPtr[i+1]; ++idx) {
            deg[i] += values[idx];
        }
    }
    cout << "【每个节点的度数】" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "节点 " << i << " 的度数: " << deg[i] << endl;
    }

    vector<vector<double>> H(n, vector<double>(inF, 1.0));
    vector<vector<double>> W(inF, vector<double>(outF, 0.1));
    W_out = W;

    AH.assign(n, vector<double>(inF, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int idx = rowPtr[i]; idx < rowPtr[i+1]; ++idx) {
            int j = colInd[idx];
            double a = values[idx] / sqrt(deg[i] * deg[j]);
            for (int f = 0; f < inF; ++f) {
                AH[i][f] += a * H[j][f];
            }
        }
    }

    vector<vector<double>> Z(n, vector<double>(outF, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < inF; ++k)
            for (int j = 0; j < outF; ++j)
                Z[i][j] += AH[i][k] * W[k][j];

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < outF; ++j)
            if (Z[i][j] < 0) Z[i][j] = 0;

    return Z;
}

// GCN 反向传播
void gcnBackward(int n, int inF, int outF, double lr, const vector<vector<double>>& AH, const vector<vector<double>>& Z, const vector<vector<double>>& Y, vector<vector<double>>& W) {
    vector<vector<double>> dZ(n, vector<double>(outF, 0.0));
    vector<vector<double>> dW(inF, vector<double>(outF, 0.0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < outF; ++j)
            dZ[i][j] = (Z[i][j] > 0) ? 2.0 * (Z[i][j] - Y[i][j]) : 0.0;

    for (int k = 0; k < inF; ++k)
        for (int j = 0; j < outF; ++j)
            for (int i = 0; i < n; ++i)
                dW[k][j] += AH[i][k] * dZ[i][j] / n;

    for (int k = 0; k < inF; ++k)
        for (int j = 0; j < outF; ++j)
            W[k][j] -= lr * dW[k][j];
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> A(n, vector<int>(n, 0));
    readGraph(n, m, A);
    printMatrix(A, "【加上自环后的邻接矩阵】：");

    cout << (isConnected(n, A) ? "【图是连通的】" : "【图不是连通的】") << endl;

    vector<int> rowPtr;
    vector<int> colInd;
    vector<double> values;
    buildCSR(n, A, rowPtr, colInd, values);

    int inF = 4, outF = 2;
    vector<vector<double>> AH, W;
    auto Z = gcnForward(n, inF, outF, rowPtr, colInd, values, AH, W);

    cout << "【GCN输出特征】:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "节点 " << i << " 的特征: ";
        for (double v : Z[i]) cout << v << " ";
        cout << endl;
    }

    // 构造 target，模拟监督信号
    vector<vector<double>> target(n, std::vector<double>(2, 0.5));
    vector<vector<double>> grad(n, vector<double>(2, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 2; ++j)
            grad[i][j] = Z[i][j] - target[i][j];  // 均方误差的梯度方向

    cout << "【反向传播误差（输出层梯度）】：" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "节点 " << i << ": ";
        for (double g : grad[i]) cout << g << " ";
        cout << endl;
    }

    return 0;
}
