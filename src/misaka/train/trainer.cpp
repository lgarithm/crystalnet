#include <misaka.h>
#include <misaka/train/agent.hpp>
#include <misaka/train/trainer.hpp>

trainer_t *new_trainer(model_t *model, operator_t *loss, optimizer_t *optimizer)
{
    return new trainer_t(model, loss, optimizer);
}

void free_trainer(trainer_t *trainer) { delete trainer; }

void run_trainer(trainer_t *trainer, dataset_t *ds) { trainer->run(*ds); }

void test_trainer(trainer_t *trainer, dataset_t *ds) { trainer->test(*ds); }

void experiment(trainer_t *trainer, dataset_t *train_ds, dataset_t *test_ds)
{
    trainer->run(*train_ds, test_ds);
}
