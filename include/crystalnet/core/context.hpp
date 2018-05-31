#pragma once
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <crystalnet/core/gc.hpp>

// A generic_context_t manages resources of a single type.
template <typename T> struct generic_context_t {
    GC<T> gc;
};

// A named_context_t manages resources of a single type with distinct names.
template <typename T> struct named_context_t : generic_context_t<T> {
    std::map<std::string, const T *> index;
    std::vector<std::pair<std::string, const T *>> items;

    explicit named_context_t(const std::string &default_prefix)
        : default_prefix(default_prefix)
    {
    }

    // own takes the ownership of a named resource.
    T *own(T *resource, const std::string &name)
    {
        if (name.empty()) {
            throw std::invalid_argument(__str__() + ": name is empty");
        }
        if (index.count(name) > 0) {
            throw std::invalid_argument(__str__() +
                                        ": duplicated name: " + name);
        }
        index[name] = resource;
        items.push_back(std::make_pair(name, resource));
        return generic_context_t<T>::gc(resource);
    }

    std::string gen_name() { return gen_name(default_prefix); }

    std::string gen_name(const std::string &prefix) const
    {
        int idx = 0;
        std::string name;
        do {
            name = prefix + std::to_string(idx);
            ++idx;
        } while (index.count(name) != 0);
        return name;
    }

  private:
    const std::string default_prefix;
    std::string __str__() const { return "context(" + default_prefix + ")"; }
};
