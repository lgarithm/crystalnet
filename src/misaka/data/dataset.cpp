#include <misaka.h>
#include <misaka/data/dataset.hpp>

// dataset_t *new_dataset() { return new dataset_t(); }

void free_dataset(dataset_t *dataset) { delete dataset; }
