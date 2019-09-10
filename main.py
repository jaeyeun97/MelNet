from config import Config
from executor import Executor


def main():
    config = Config()
    executor = Executor(config)

    if config.mode == 'train':
        executor.train()
    elif config.mode == 'sample':
        executor.sample()
    else:
        executor.test(config.mode)


if __name__ == "__main__": 
    main()
