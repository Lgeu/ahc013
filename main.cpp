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

namespace dence {

// 蜜なパターン
// ビームサーチ
template <int N> void Solve() {
    //
}
} // namespace dence

namespace input {
int N, K;
}

using Point = Vec2<signed char>;
struct Move {
    Point from, to;
    inline void Print() const {
        cout << (int)from.y << ' ' << (int)from.x << ' ' << (int)to.y << ' '
             << (int)to.x << endl;
    }
};

namespace sparse {

// 疎なパターン
// 最初に横を開通させる

template <int N> struct State {
    Board<signed char, N, N> board;
    void Read() {
        for (auto y = 0; y < N; y++) {
            string s;
            cin >> s;
            for (auto x = 0; x < N; x++) {
                board[{y, x}] = s[x] - '0';
            }
        }
    }

    array<Move, 500> moves;
    int n_moves = 0;

    array<Move, 500> connections;
    int n_connections = 0;

    void Apply(Move pp) {
        assert(board[pp.from] != 0);
        assert(board[pp.to] == 0);
        board[pp.to] = board[pp.from];
        board[pp.from] = 0;
        moves[n_moves++] = pp;
    }

    inline int RemainingMoves() const {
        return input::K * 100 - n_moves - n_connections;
    }

    inline void Print() const {
        cout << n_moves << endl;
        for (int i = 0; i < n_moves; i++)
            moves[i].Print();
        cout << n_connections << endl;
        for (int i = 0; i < n_connections; i++)
            connections[i].Print();
    }
};

template <int N> void Solve() {
    auto state = State<N>();
    state.Read();

    auto& board = state.board;

    // 上を揃える
    {
        const auto cy = 1;
        for (auto x = 0; x < N; x++) {
            const auto c = Point(cy, x);
            const auto u = Point(cy - 1, x);
            const auto d = Point(cy + 1, x);
            const auto dd = Point(cy + 2, x);
            const auto ul = Point(cy - 1, max(0, x - 1));
            const auto dl = Point(cy + 1, max(0, x - 1));
            const auto ur = Point(cy - 1, min(N - 1, x + 1));
            const auto dr = Point(cy + 1, min(N - 1, x + 1));
            const auto ddd = Point(cy + 3, x);
            const auto ddl = Point(cy + 2, max(0, x - 1));
            const auto ull = Point(cy - 1, max(0, x - 2));
            const auto dll = Point(cy + 1, max(0, x - 2));
            const auto ddr = Point(cy + 2, min(N - 1, x + 1));
            const auto urr = Point(cy - 1, min(N - 1, x + 2));
            const auto drr = Point(cy + 1, min(N - 1, x + 2));
            const auto dddd = Point(cy + 4, x);
            const auto dddl = Point(cy + 3, max(0, x - 1));
            const auto ddll = Point(cy + 2, max(0, x - 2));
            const auto ulll = Point(cy - 1, max(0, x - 3));
            const auto dlll = Point(cy + 1, max(0, x - 3));

            switch (board[c]) {
            case 0:
                if (board[d] == 1)
                    state.Apply({d, c});
                else if (board[u] == 1)
                    state.Apply({u, c});
                break;
            case 1:
                break; // do nothing
            default:
                if (board[u] == 0)
                    state.Apply({c, u});
                else if (board[d] == 0)
                    state.Apply({c, d});
                else if (board[dd] == 0) {
                    state.Apply({d, dd});
                    state.Apply({c, d});
                } else if (board[ul] == 0) {
                    state.Apply({u, ul});
                    state.Apply({c, u});
                } else if (board[dl] == 0) {
                    state.Apply({d, dl});
                    state.Apply({c, d});
                } else if (board[ur] == 0) {
                    state.Apply({u, ur});
                    state.Apply({c, u});
                } else if (board[dr] == 0) {
                    state.Apply({d, dr});
                    state.Apply({c, d});
                } else if (board[ddd] == 0) {
                    state.Apply({dd, ddd});
                    state.Apply({d, dd});
                    state.Apply({c, d});
                } else if (board[ddl] == 0) {
                    state.Apply({dd, ddl});
                    state.Apply({d, dd});
                    state.Apply({c, d});
                } else if (board[ull] == 0) {
                    state.Apply({ul, ull});
                    state.Apply({u, ul});
                    state.Apply({c, u});
                } else if (board[dll] == 0) {
                    state.Apply({dl, dll});
                    state.Apply({d, dl});
                    state.Apply({c, d});
                } else if (board[ddr] == 0) {
                    state.Apply({dd, ddr});
                    state.Apply({d, dd});
                    state.Apply({c, d});
                } else if (board[urr] == 0) {
                    state.Apply({ur, urr});
                    state.Apply({u, ur});
                    state.Apply({c, u});
                } else if (board[drr] == 0) {
                    state.Apply({dr, drr});
                    state.Apply({d, dr});
                    state.Apply({c, d});
                } else if (board[dddd] == 0) {
                    state.Apply({ddd, dddd});
                    state.Apply({dd, ddd});
                    state.Apply({d, dd});
                    state.Apply({c, d});
                } else if (board[dddl] == 0) {
                    state.Apply({ddd, dddl});
                    state.Apply({dd, ddd});
                    state.Apply({d, dd});
                    state.Apply({c, d});
                } else if (board[ddll] == 0) {
                    state.Apply({ddl, ddll});
                    state.Apply({dd, ddl});
                    state.Apply({d, dd});
                    state.Apply({c, d});
                } else if (board[ulll] == 0) {
                    state.Apply({ull, ulll});
                    state.Apply({ul, ull});
                    state.Apply({u, ul});
                    state.Apply({c, u});
                } else if (board[dlll] == 0) {
                    state.Apply({dll, dlll});
                    state.Apply({dl, dll});
                    state.Apply({d, dl});
                    state.Apply({c, d});
                } else {
                    assert(false);
                }

                if (board[d] == 1)
                    state.Apply({d, c});
                else if (board[u] == 1)
                    state.Apply({u, c});

                // TODO
                break;
            }
        }
    }
    state.Print();
}
} // namespace sparse

void Solve() {
    cin >> input::N >> input::K;
    switch (input::N) {
        // clang-format off

        #define CASE(n) case n: sparse::Solve<n>(); break;
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

// 最後に↓を貼る
#ifdef __GNUC__
#pragma clang attribute pop
#endif
// 最後に↑を貼る