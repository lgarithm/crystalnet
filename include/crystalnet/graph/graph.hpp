#pragma once
#include <set>
#include <vector>

template <typename T> struct graph_visitor {
    std::set<const T *> visited;
    std::vector<const T *> list;

    void pre_order(const T *n)
    {
        if (visited.count(n) > 0) { return; }
        visited.insert(n);
        list.push_back(n);
        for (const T *next : n->predecessors()) { pre_order(next); }
    }

    void post_order(const T *n)
    {
        if (visited.count(n) > 0) { return; }
        visited.insert(n);
        for (const T *next : n->predecessors()) { post_order(next); }
        list.push_back(n);
    }
};

template <typename T> std::vector<const T *> traverse_pre(const T *n)
{
    graph_visitor<T> visit;
    visit.pre_order(n);
    return visit.list;
}

template <typename T> std::vector<const T *> traverse_post(const T *n)
{
    graph_visitor<T> visit;
    visit.post_order(n);
    return visit.list;
}
