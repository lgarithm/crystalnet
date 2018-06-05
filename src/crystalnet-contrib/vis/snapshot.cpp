#include <crystalnet-contrib/vis/snapshot.hpp>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/core/user_context.hpp>
#include <crystalnet/model/model.hpp>
#include <crystalnet/symbol/model.hpp>

void save_layers(const model_t &m, const s_model_t &s)
{
    int idx = 0;
    for (const auto l : s.ctx._layers.items) {
        const auto node = m.ctx.index.at(l->name);
        char name[260];
        sprintf(name, "layer%04d.idx", idx++);
        const auto t = node->value();
        save_tensor(name, &t);
        logf("saved to %s", name);
    }
}
