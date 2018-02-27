#include <misaka.h>
#include <misaka/data/dataset.hpp>

// dataset_t *new_dataset() { return new dataset_t(); }

void free_dataset(dataset_t *dataset) { delete dataset; }
const shape_t *ds_image_shape(dataset_t *ds) { return ds->image_shape(); }
const shape_t *ds_label_shape(dataset_t *ds) { return ds->label_shape(); }
