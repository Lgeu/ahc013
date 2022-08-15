#pragma once

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

//#include <atcoder/all>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __GNUC__
#include <x86intrin.h>
#endif

#ifdef __clang__
#pragma clang attribute push(__attribute__((target("arch=skylake"))),          \
                             apply_to = function)

#elif defined(__GNUC__)
#pragma GCC target(                                                            \
    "sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")
#pragma GCC optimize("Ofast")
#endif

// ========================== macroes ==========================

//#define NDEBUG

#define rep(i, n) for (auto i = 0; (i) < (n); (i)++)
#define rep1(i, n) for (auto i = 1; (i) <= (n); (i)++)
#define rep3(i, s, n) for (auto i = (s); (i) < (n); (i)++)

#define CHECK(var)                                                             \
    do {                                                                       \
        std::cout << #var << '=' << var << endl;                               \
    } while (false)

// ========================== utils ==========================

using namespace std;
using ll = long long;
constexpr double PI = 3.1415926535897932;

template <class T, class S> inline bool chmin(T& m, S q) {
    if (m > q) {
        m = q;
        return true;
    } else
        return false;
}

template <class T, class S> inline bool chmax(T& m, const S q) {
    if (m < q) {
        m = q;
        return true;
    } else
        return false;
}

// 2 次元ベクトル
template <typename T> struct Vec2 {
    /*
    y 軸正は下方向
    x 軸正は右方向
    回転は時計回りが正（y 軸正を上と考えると反時計回りになる）
    */
    using value_type = T;
    T y, x;
    constexpr inline Vec2() = default;
    constexpr Vec2(const T& arg_y, const T& arg_x) : y(arg_y), x(arg_x) {}
    inline Vec2(const Vec2&) = default;            // コピー
    inline Vec2(Vec2&&) = default;                 // ムーブ
    inline Vec2& operator=(const Vec2&) = default; // 代入
    inline Vec2& operator=(Vec2&&) = default;      // ムーブ代入
    template <typename S>
    constexpr inline Vec2(const Vec2<S>& v) : y((T)v.y), x((T)v.x) {}
    inline Vec2 operator+(const Vec2& rhs) const {
        return Vec2(y + rhs.y, x + rhs.x);
    }
    inline Vec2 operator+(const T& rhs) const { return Vec2(y + rhs, x + rhs); }
    inline Vec2 operator-(const Vec2& rhs) const {
        return Vec2(y - rhs.y, x - rhs.x);
    }
    template <typename S> inline Vec2 operator*(const S& rhs) const {
        return Vec2(y * rhs, x * rhs);
    }
    inline Vec2 operator*(const Vec2& rhs) const { // x + yj とみなす
        return Vec2(x * rhs.y + y * rhs.x, x * rhs.x - y * rhs.y);
    }
    template <typename S> inline Vec2 operator/(const S& rhs) const {
        assert(rhs != 0.0);
        return Vec2(y / rhs, x / rhs);
    }
    inline Vec2 operator/(const Vec2& rhs) const { // x + yj とみなす
        return (*this) * rhs.inv();
    }
    inline Vec2& operator+=(const Vec2& rhs) {
        y += rhs.y;
        x += rhs.x;
        return *this;
    }
    inline Vec2& operator-=(const Vec2& rhs) {
        y -= rhs.y;
        x -= rhs.x;
        return *this;
    }
    template <typename S> inline Vec2& operator*=(const S& rhs) const {
        y *= rhs;
        x *= rhs;
        return *this;
    }
    inline Vec2& operator*=(const Vec2& rhs) { return *this = (*this) * rhs; }
    inline Vec2& operator/=(const Vec2& rhs) { return *this = (*this) / rhs; }
    inline bool operator!=(const Vec2& rhs) const {
        return x != rhs.x || y != rhs.y;
    }
    inline bool operator==(const Vec2& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
    inline void rotate(const double& rad) { *this = rotated(rad); }
    inline Vec2<double> rotated(const double& rad) const {
        return (*this) * rotation(rad);
    }
    static inline Vec2<double> rotation(const double& rad) {
        return Vec2(sin(rad), cos(rad));
    }
    static inline Vec2<double> rotation_deg(const double& deg) {
        return rotation(PI * deg / 180.0);
    }
    inline Vec2<double> rounded() const {
        return Vec2<double>(round(y), round(x));
    }
    inline Vec2<double> inv() const { // x + yj とみなす
        const double norm_sq = l2_norm_square();
        assert(norm_sq != 0.0);
        return Vec2(-y / norm_sq, x / norm_sq);
    }
    inline double l2_norm() const { return sqrt(x * x + y * y); }
    inline double l2_norm_square() const { return x * x + y * y; }
    inline T l1_norm() const { return std::abs(x) + std::abs(y); }
    inline double abs() const { return l2_norm(); }
    inline double phase() const { // [-PI, PI) のはず
        return atan2(y, x);
    }
    inline double phase_deg() const { // [-180, 180) のはず
        return phase() * (180.0 / PI);
    }
};
template <typename T, typename S>
inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) {
    return rhs * lhs;
}
template <typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
    os << vec.y << ' ' << vec.x;
    return os;
}

// 2 次元配列
template <class T, int height, int width> struct Board {
    array<T, height * width> data;
    template <class Int> constexpr inline auto& operator[](const Vec2<Int>& p) {
        return data[width * p.y + p.x];
    }
    template <class Int>
    constexpr inline const auto& operator[](const Vec2<Int>& p) const {
        return data[width * p.y + p.x];
    }
    template <class Int>
    constexpr inline auto& operator[](const initializer_list<Int>& p) {
        return data[width * *p.begin() + *(p.begin() + 1)];
    }
    template <class Int>
    constexpr inline const auto&
    operator[](const initializer_list<Int>& p) const {
        return data[width * *p.begin() + *(p.begin() + 1)];
    }
    constexpr inline void Fill(const T& fill_value) {
        fill(data.begin(), data.end(), fill_value);
    }
    void Print() const {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cout << data[width * y + x] << " \n"[x == width - 1];
            }
        }
    }
};

// 時間 (秒)
inline double Time() {
    return static_cast<double>(
               chrono::duration_cast<chrono::nanoseconds>(
                   chrono::steady_clock::now().time_since_epoch())
                   .count()) *
           1e-9;
}

// ======================================================================

auto rng = mt19937(42);

namespace input {
int N, K;
}

using Point = Vec2<signed char>;
struct alignas(4) Move {
    Point from, to;
    inline void Print() const {
        cout << (int)from.y << ' ' << (int)from.x << ' ' << (int)to.y << ' '
             << (int)to.x << endl;
    }
    inline bool Empty() const {
        return from.y == 0 && from.x == 0 && to.y == 0 && to.x == 0;
    }
    inline void Reset() {
        from.y = 0;
        from.x = 0;
        to.y = 0;
        to.x = 0;
    }
};

struct Cell {
    signed char color;
    signed char index;
};

template <int N> struct Input {
    Board<Cell, N, N> board;
    array<array<Point, 100>, 6> where;
    void Read() {
        auto number_counts = array<int, 6>();
        for (auto y = 0; y < N; y++) {
            string s;
            cin >> s;
            for (auto x = 0; x < N; x++) {
                int c = s[x] - '0';
                board[{y, x}] = {(signed char)c, (signed char)number_counts[c]};
                if (c)
                    where[c][number_counts[c]++] = {(signed char)y,
                                                    (signed char)x};
            }
        }
        assert(number_counts[1] == 100);
        assert(number_counts[2] == 100);
    }
};

// from https://atcoder.jp/contests/practice2/submissions/16581325
struct UnionFind {
    array<signed char, 100> data;
    inline UnionFind() { fill(data.begin(), data.end(), (signed char)-1); }
    inline signed char find(const signed char k) {
        return data[k] < 0 ? k : data[k] = find(data[k]);
    }
    inline int unite(signed char x, signed char y) {
        if ((x = find(x)) == (y = find(y)))
            return false;
        if (data[x] > data[y])
            swap(x, y);
        data[x] += data[y];
        data[y] = x;
        return true;
    }
    inline int size(signed char k) { return -data[find(k)]; }
    inline int same(signed char x, signed char y) { return find(x) == find(y); }
};

template <int N_, int K_> struct State {
    static constexpr auto N = N_;
    static constexpr auto K = K_;
    Board<Cell, N, N> board; // 最終状態
    array<array<Point, 100>, 2> where12;
    double score;

    array<UnionFind, 2> ufs;

    array<Move, K * 100> moves;
    int n_moves;

    array<Move, K * 100> connections;
    int n_connections;

    State()
        : board(), where12(), score(), ufs(), moves(), n_moves(), connections(),
          n_connections() {}

    inline int RemainingMoves() const {
        return K * 100 - n_moves - n_connections;
    }

    inline void Print() const {
        cout << n_moves << endl;
        for (int i = 0; i < K * 100; i++)
            if (!moves[i].Empty())
                moves[i].Print();
        cout << n_connections << endl;
        for (int i = 0; i < K * 100; i++)
            if (!connections[i].Empty())
                connections[i].Print();
    }

    inline bool Movable(Move pp) const {
        return board[pp.from].color != 0 && board[pp.to].color == 0;
    }

    inline void ApplyMove(Move pp) {
        assert(board[pp.from].color != 0);
        assert(board[pp.to].color == 0);
        board[pp.to] = board[pp.from];
        board[pp.from] = {};
        if (board[pp.to].color <= 2)
            where12[board[pp.to].color - 1][board[pp.to].index] = pp.to;
    }

    inline bool Connect(Move pp, bool no_check = false) {
        const auto index_from = board[pp.from].index;
        const auto index_to = board[pp.to].index;

        if (!no_check) {
            // 右か下を仮定
            if (board[pp.from].color != board[pp.to].color)
                return false;
            if (board[pp.from].color != 1 && board[pp.from].color != 2)
                return false;
            auto& uf = ufs[board[pp.from].color - 1];
            if (uf.same(index_from, index_to))
                return false;
            if (pp.from.y == pp.to.y) {
                for (auto x = pp.from.x + 1; x < pp.to.x; x++)
                    if (board[{(int)pp.from.y, x}].color != 0)
                        return false;
            } else {
                assert(pp.from.x == pp.to.x);
                for (auto y = pp.from.y + 1; y < pp.to.y; y++)
                    if (board[{y, (int)pp.from.x}].color != 0)
                        return false;
            }
        }

        // ここまで来れば接続可能
        {
            auto& uf = ufs[board[pp.from].color - 1];
            if (pp.from.y == pp.to.y) {
                for (auto x = pp.from.x + 1; x < pp.to.x; x++)
                    board[{(int)pp.from.y, x}].color = 9;
            } else {
                assert(pp.from.x == pp.to.x);
                for (auto y = pp.from.y + 1; y < pp.to.y; y++)
                    board[{y, (int)pp.from.x}].color = 9;
            }
            score += uf.size(index_from) * uf.size(index_to);
            uf.unite(index_from, index_to);
        }
        return true;
    }

    void RandomUpdate(const Input<N>& initial_board) {
        // inplace で処理する
        // ランダムで 1 手消去する
        // 空いていたら、1/2 で手を追加

        static constexpr auto kRadius = 3;

        static constexpr auto kCheckPerf = true;
        static auto t_move = 0.0;
        static auto t_conn_1 = 0.0;
        static auto t_conn_2 = 0.0;
        static auto t0 = 0.0;
        static auto iteration = 0;
        if constexpr (kCheckPerf) {
            t0 = Time();
            iteration++;
        }

        board = initial_board.board;
        where12[0] = initial_board.where[1];
        where12[1] = initial_board.where[2];
        ufs[0] = UnionFind();
        ufs[1] = UnionFind();
        score = 0.0;

        assert(n_moves + n_connections <= K * 100);
        auto steps_erase = array<int, 2>();
        for (auto& s : steps_erase)
            s = n_moves + n_connections == 0
                    ? -1
                    : uniform_int_distribution<>(0, n_moves + n_connections -
                                                        1)(rng);
        auto nth_move = 0;
        auto empty_indices = array<short, K * 100>(); // 後ろはconnection
        auto n_empty_indices = 0;
        auto n_seen_connections = 0;

        // 変化させる位置を決める
        // これ毎回変えるべきか……？
        auto target_center = Point();
        {
            const auto target_num = uniform_int_distribution<>(0, 1)(rng);
            auto mi = 101;
            for (auto trial = 0; trial < 5; trial++) {
                const auto index = uniform_int_distribution<>(0, 99)(rng);
                const auto siz = ufs[target_num].size(index);
                if (chmin(mi, siz)) {
                    target_center = where12[target_num][index];
                }
            }
        }

        for (auto i = 0; i < K * 100; i++) {
            if (!moves[i].Empty()) {
                if (nth_move == steps_erase[0] || nth_move == steps_erase[1]) {
                    moves[i].Reset();
                    n_moves--;
                }
                nth_move++;
            } else if (!connections[i].Empty()) {
                if (nth_move == steps_erase[0] || nth_move == steps_erase[1]) {
                    connections[i].Reset();
                    n_connections--;
                }
                nth_move++;
            }

            // 移動不可なら取り除く
            if (!moves[i].Empty() && !Movable(moves[i])) {
                moves[i].Reset();
                n_moves--;
            }

            // メモ
            if (!connections[i].Empty()) {
                empty_indices[100 * K - ++n_seen_connections] = i;
            }

            if (moves[i].Empty() && connections[i].Empty()) {
                // 1/2 でスキップ
                const auto rn = uniform_real_distribution<>()(rng);
                if (rn < 0.5) { // パラメータ
                    empty_indices[n_empty_indices++] = i;
                    continue;
                }

                // 移動を追加
                auto trial = 0;
                do {
                    if (trial >= 50) {
                        empty_indices[n_empty_indices++] = i;
                        break;
                    }
                    trial++;
                    const auto rc = uniform_int_distribution<>(0, 1)(rng);
                    if (rc == 0) {
                        // 横
                        // const auto y =
                        //     uniform_int_distribution<>(0, N - 1)(rng);
                        // const auto x =
                        //     uniform_int_distribution<>(0, N - 2)(rng);
                        const auto y =
                            target_center.y +
                            uniform_int_distribution<>(0, kRadius)(rng) -
                            uniform_int_distribution<>(0, kRadius)(rng);
                        if (y < 0 || N <= y)
                            continue;
                        const auto x =
                            target_center.x +
                            uniform_int_distribution<>(0, kRadius - 1)(rng) -
                            uniform_int_distribution<>(0, kRadius)(rng);
                        if (x < 0 || N - 1 <= x)
                            continue;
                        const auto l = Point(y, x);
                        const auto r = Point(y, x + 1);
                        if ((board[l].color == 0) == (board[r].color == 0))
                            continue;
                        if (board[l].color == 0) {
                            // ←
                            moves[i] = {r, l};
                        } else {
                            // →
                            moves[i] = {l, r};
                        }
                    } else {
                        // 縦
                        // const auto x =
                        //     uniform_int_distribution<>(0, N - 1)(rng);
                        // const auto y =
                        //     uniform_int_distribution<>(0, N - 2)(rng);
                        const auto x =
                            target_center.x +
                            uniform_int_distribution<>(0, kRadius)(rng) -
                            uniform_int_distribution<>(0, kRadius)(rng);
                        if (x < 0 || N <= x)
                            continue;
                        const auto y =
                            target_center.y +
                            uniform_int_distribution<>(0, kRadius - 1)(rng) -
                            uniform_int_distribution<>(0, kRadius)(rng);
                        if (y < 0 || N - 1 <= y)
                            continue;
                        const auto u = Point(y, x);
                        const auto d = Point(y + 1, x);
                        if ((board[u].color == 0) == (board[d].color == 0))
                            continue;
                        if (board[u].color == 0) {
                            // ↑
                            moves[i] = {d, u};
                        } else {
                            // ↓
                            moves[i] = {u, d};
                        }
                    }
                    n_moves++;
                    break;
                } while (true);
            }

            // 盤面を更新
            if (!moves[i].Empty())
                ApplyMove(moves[i]);
        }

        if constexpr (kCheckPerf) {
            const auto t1 = Time();
            t_move += t1 - t0;
            t0 = t1;
        }

        assert(n_seen_connections == n_connections);
        assert(n_moves + n_connections <= K * 100);

        // connection
        // メモ:
        // ↑でconnectionが消されてmoveが追加されない場合、かならずConnectは成功する
        for (auto idx_empty_indices = K * 100 - n_connections;
             idx_empty_indices < K * 100; idx_empty_indices++) {
            const auto i = empty_indices[idx_empty_indices];
            assert(!connections[i].Empty());
            assert(moves[i].Empty());
            if (!Connect(connections[i])) {
                connections[i].Reset();
                n_connections--;
                empty_indices[n_empty_indices++] = i;
            }
        }

        if constexpr (kCheckPerf) {
            const auto t1 = Time();
            t_conn_1 += t1 - t0;
            t0 = t1;
        }

        // 貪欲で繋ぐ
        // メモ: 後から繋ぐので、繋ぐことで他の繋ぎが不可能になることはない
        auto idx_empty_indices = 0;
        for (auto target = 1; target <= 2; target++) {
            auto order = array<signed char, 100>();
            iota(order.begin(), order.end(), 0);
            shuffle(order.begin(), order.end(), rng);
            auto idx_order = 0;
            auto& uf = ufs[target - 1];
            const auto& where = where12[target - 1];

            for (; idx_empty_indices < n_empty_indices; idx_empty_indices++) {
                const auto i = empty_indices[idx_empty_indices];
                assert(moves[i].Empty() && connections[i].Empty());
                while (idx_order < 100) {
                    const auto from = where[order[idx_order++]];
                    // 横
                    for (auto to = Point(from.y, (signed char)(from.x + 1));
                         to.x < N; to.x++) {
                        if (board[to].color == target &&
                            !uf.same(board[to].index, board[from].index)) {
                            connections[i] = {from, to};
                            const auto con = Connect(connections[i], true);
                            assert(con);
                            n_connections++;
                            goto connected;
                        } else if (board[to].color != 0) {
                            break;
                        }
                    }
                    // 縦
                    for (auto to = Point((signed char)(from.y + 1), from.x);
                         to.y < N; to.y++) {
                        if (board[to].color == target &&
                            !uf.same(board[to].index, board[from].index)) {
                            connections[i] = {from, to};
                            const auto con = Connect(connections[i], true);
                            assert(con);
                            n_connections++;
                            goto connected;
                        } else if (board[to].color != 0) {
                            break;
                        }
                    }
                    // 次の from 候補へ
                }
                // from 候補がなくなった
                break;

            connected:;
            }
        }

        if constexpr (kCheckPerf) {
            const auto t1 = Time();
            t_conn_2 += t1 - t0;
            t0 = t1;

            if (iteration % 50000 == 0) {
                // cerr << "t_move,t_conn_1,t_conn_2=";
                // cerr << t_move << "," << t_conn_1 << "," << t_conn_2 << endl;
            }
        }
    }

    void SanityCheck() const {
        auto n_moves_count = 0;
        for (auto pp : moves) {
            if (!pp.Empty())
                n_moves_count++;
        }
        assert(n_moves == n_moves_count);
    }
};

inline double sigmoid(const double& a, const double& x) {
    return 1.0 / (1.0 + exp(-a * x));
}

// f: [0, 1] -> [0, 1]
inline double MonotonicallyIncreasingFunction(const double a, const double b,
                                              const double x) {
    if (a == 0.0)
        return x;
    const double x_left = a > 0 ? -b - 0.5 : b - 0.5;
    const double x_right = x_left + 1.0;
    const double left = sigmoid(a, x_left);
    const double right = sigmoid(a, x_right);
    const double y = sigmoid(a, x + x_left);
    return (y - left) /
           (right - left); // left とかが大きい値になると誤差がヤバイ　最悪 0
                           // 除算になる  // b が正なら大丈夫っぽい
}

// f: [0, 1] -> [start, end]
inline double MonotonicFunction(const double start, const double end,
                                const double a, const double b,
                                const double x) {
    return MonotonicallyIncreasingFunction(a, b, x) * (end - start) + start;
}

inline double Temperature(const double t) {
    // return MonotonicFunction(20.0, 2.0, -3.0, 2.0, t);
    return MonotonicFunction(5.0, 0.5, 0.0, 0.0, t);
}

template <int N, int K> void SolveN() {
    auto input = Input<N>();
    input.Read();
    auto state = State<N, K>();

    auto t0 = Time();
    auto iteration = 0;
    static constexpr auto kTimeLimit = 2.9;

    while (true) {
        const auto t = Time() - t0;
        if (t > kTimeLimit)
            break;
        const auto progress_rate = t / kTimeLimit;
        auto updated_state = state;
        updated_state.RandomUpdate(input);
        const auto gain = updated_state.score - state.score;
        const auto temperature = Temperature(progress_rate);
        const auto acceptance_proba = exp(gain / temperature);
        if (uniform_real_distribution<>()(rng) < acceptance_proba) {
            state = updated_state;
        }

        iteration++;
        // if (iteration % 10000 == 0)
        //     state.Print();
    }

    state.Print();
    cerr << iteration << " iterations" << endl;
}

void Solve() {
    cin >> input::N >> input::K;
    switch (input::N) {
        // clang-format off

        #define CASE(n) case n: switch(input::K) { case 2: SolveN<n, 2>(); break; case 3: SolveN<n, 3>(); break; case 4: SolveN<n, 4>(); break; case 5: SolveN<n, 5>(); break; } break;
                                                     CASE(15) CASE(16) CASE(17) CASE(18) CASE(19)
        CASE(20) CASE(21) CASE(22) CASE(23) CASE(24) CASE(25) CASE(26) CASE(27) CASE(28) CASE(29)
        CASE(30) CASE(31) CASE(32) CASE(33) CASE(34) CASE(35) CASE(36) CASE(37) CASE(38) CASE(39)
        CASE(40) CASE(41) CASE(42) CASE(43) CASE(44) CASE(45) CASE(46) CASE(47) CASE(48)
        // clang-format on
    }
}

int main() {
    //
    Solve();
}

#ifdef __GNUC__
#pragma clang attribute pop
#endif

// python3 score_all.py
// flamegraph -- ./a.out < ./tools/in/0011.txt
// g++ main.cpp -Wall -Wextra -std=c++17 -g -O2