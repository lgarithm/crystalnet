#pragma onece

#include <cstdint>
#include <tuple>
#include <utility>

template <typename T> struct enumerate_t {
    const T &e;

    explicit enumerate_t(const T &e) : e(e) {}

    template <typename I> struct iter_t {
        uint32_t idx;
        I item;
        iter_t(uint32_t idx, const I &item) : idx(idx), item(item) {}

        bool operator!=(const iter_t &it) const { return item != it.item; }

        void operator++()
        {
            ++idx;
            ++item;
        }
        auto operator*() const { return std::make_pair(idx, *item); }
    };

    template <typename I> static iter_t<I> iter(uint32_t idx, const I &item)
    {
        return iter_t<I>(idx, item);
    }

    auto begin() const { return iter(0, e.begin()); }
    auto end() const { return iter((uint32_t)-1, e.end()); }
};

template <typename T> enumerate_t<T> enumerate(const T &e)
{
    return enumerate_t<T>(e);
}
