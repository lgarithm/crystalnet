#include <crystalnet.h>
#include <crystalnet/train/agent.hpp>
#include <crystalnet/train/trainer.hpp>

s_trainer_t *new_s_trainer(s_model_t *model, operator_t *loss,
                           optimizer_t *optimizer)
{
    return new s_trainer_t(model, loss, optimizer);
}

void del_s_trainer(s_trainer_t *trainer) { delete trainer; }

void s_experiment(s_trainer_t *trainer, dataset_t *train_ds, dataset_t *test_ds,
                  uint32_t batch_size)
{
    trainer->run(*train_ds, test_ds, batch_size);
}

void s_trainer_run(s_trainer_t *trainer, dataset_t *ds) { trainer->run(*ds); }
