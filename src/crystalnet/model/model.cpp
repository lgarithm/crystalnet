#include <queue>
#include <set>
#include <string>
#include <vector>

#include <crystalnet/graph/graph.hpp>
#include <crystalnet/model/model.hpp>

void model_t::forward() const { return ::forward(&output); }

void model_t::backward() const { return ::backward(&output); }

void forward(const node_t *output)
{
    const auto list = traverse_post(output);
    for (const node_t *n : list) { n->forward_eval(); }
}

void backward(const node_t *output)
{
    const auto list = traverse_pre(output);
    for (const auto &n : list) { n->backward_eval(); }
}
