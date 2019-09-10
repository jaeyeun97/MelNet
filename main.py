from config import Config
from data import get_dataloader
from model import MelNetModel

config = Config()
model = MelNetModel(config)

if config.mode != 'sample':
    dataloader = get_dataloader(config)

    i = 1
    for epoch in range(1, config.epochs + 1):
        for x in dataloader:
            losses = model.step(x)
            print("Iteration %s:\t%s" % (i, '\t'.join(str(l) for l in losses)))

            if i % config.iter_interval == 0:
                print(f"Storing network for iter {i}")
                model.save_networks(it=i)
            i += 1

        if epoch % config.epoch_interval == 0:
            print(f"Storing network for epoch {epoch}")
            model.save_networks(epoch=epoch)
