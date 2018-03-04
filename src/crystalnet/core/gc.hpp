#pragma once
#include <memory>
#include <vector>

template <typename T> struct GC {
    using owner_t = std::unique_ptr<T>;
    std::vector<owner_t> allocs;

    T *operator()(T *p)
    {
        allocs.push_back(owner_t(p));
        return p;
    }

    // static GC default_gc;
    // static T *gc(T *p) { return default_gc(p); }
};

// TODO: provide a generic gc function
// template <typename T> T *gc0(T *p) { return GC<T>::gc(p); }

template <typename T> struct Ref {
    std::vector<T *> items;

    T *operator()(T *p)
    {
        items.push_back(p);
        return p;
    }
};
