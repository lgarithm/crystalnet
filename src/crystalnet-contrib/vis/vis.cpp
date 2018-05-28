#include <crystalnet.h>
#include <crystalnet/symbol/model.hpp>

#include <cstdio>

void graphviz(const s_model_t &m, FILE *fp)
{

    fprintf(fp, "digraph{\n");
    {
        fprintf(fp, "\t// placeholder nodes\n");
        for (const auto &node : m.ctx.places.items) {
            fprintf(fp, "\t\"%s\" [shape=box];\n", node->name.c_str());
        }
    }
    {
        fprintf(fp, "\t// parameter nodes\n");
        for (const auto &node : m.ctx.params.items) {
            fprintf(fp, "\t\"%s\" [shape=box, style=filled, fillcolor=grey];\n",
                    node->name.c_str());
        }
    }
    {
        fprintf(fp, "\t// operator nodes\n");
        for (const auto &node : m.ctx.ops.items) {
            fprintf(fp, "\t\"%s\";\n", node->name.c_str());
        }
    }
    fprintf(fp, "\n");
    {
        fprintf(fp, "\t// edges\n");
        for (const auto &node : m.ctx.ops.items) {
            for (const auto &prev : node->inputs.nodes) {
                fprintf(fp, "\t\"%s\" -> \"%s\";\n",  //
                        prev->name.c_str(), node->name.c_str());
            }
        }
    }
    fprintf(fp, "}\n");
}
