#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <queue>

#include <crystalnet/core/error.hpp>

struct MB {
    size_t size;
    std::unique_ptr<uint8_t> data;
    explicit MB(size_t size) : size(size), data(new uint8_t[size]) {}
};

struct MP {
    size_t allocated;
    size_t available;
    std::map<size_t, std::queue<MB *>> allocs;

    MP() : allocated(0), available(0) { check(false); }

    ~MP()
    {
        for (auto[_, q] : allocs) {
            while (!q.empty()) {
                delete q.front();
                q.pop();
            }
        }
    }

    MB *get(size_t size)
    {
        auto &q = allocs[size];
        if (q.empty()) {
            q.push(new MB(size));
            allocated += size;
            available += size;
        }
        auto mb = q.front();
        q.pop();
        available -= size;
        return mb;
    }

    void put(MB *mb)
    {
        size_t size = mb->size;
        allocs[size].push(mb);
        available += size;
    }
};
