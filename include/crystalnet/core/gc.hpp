#pragma once
#include <memory>
#include <vector>

template <typename T> struct GC {
    T *operator()(T *p)
    {
        allocs.push_back(owner_t(p));
        return p;
    }

  private:
    using owner_t = std::unique_ptr<T>;
    std::vector<owner_t> allocs;
};

template <typename T> T *gc(T *p)
{
    static GC<T> _gc;
    return _gc(p);
}

template <typename T> struct Ref {
    std::vector<T *> items;

    T *operator()(T *p)
    {
        items.push_back(p);
        return p;
    }
};
