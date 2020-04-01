import torch

from copy import deepcopy
from threading import Thread, Event
from queue import PriorityQueue, Empty


def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.set_worker(worker_id)


class DataLoader(object):
    def __init__(self, dataset, num_workers, batch_size):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      num_workers=num_workers,
                                                      worker_init_fn=_worker_init_fn,
                                                      batch_size=None,
                                                      batch_sampler=None)

    def __iter__(self):
        if self.num_workers > 0:
            loader_iter = _ParallelDataLoaderIter(self.dataloader,
                                                  self.num_workers,
                                                  self.batch_size)
        elif self.num_workers == 0:
            loader_iter =  _SeqDataLoaderIter(self.dataloader,
                                              self.batch_size)
        else:
            raise ValueError('``num_workers`` need to be positive.')
        return _DataPipeline(loader_iter)


def _collate(data):
    def collate(items):
        if type(items[0]) == torch.Tensor:
            return torch.stack(items)
        else:
            return items
    return tuple(collate(items) for items in zip(*data))


class _DataPipeline(object):
    def __init__(self, loader_iter):
        self.loader_iter = loader_iter
        self.next_item = None
        self.stream = torch.cuda.Stream()
        self._prefetch()

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        item = self.next_item

        if item is None:
            raise StopIteration

        item[1].record_stream(torch.cuda.current_stream())
        self._prefetch()
        return item

    def _prefetch(self):
        try:
            self.next_item = _collate(next(self.loader_iter))[1:]
        except StopIteration:
            self.next_item = None
            return

        with torch.cuda.stream(self.stream):
            indicies, block, *rest = self.next_item
            self.next_item = (indicies, block.cuda(non_blocking=True), *rest)


class _BaseDataLoaderIter(object):
    def __init__(self, dataloader, batch_size):
        self.dataloader = dataloader
        self.batch_size = batch_size

    def __iter__(self):
        return self


def _dataload_thread(dataloader, pq, event):
    event.set()
    for item in dataloader:
        # add to priority queue based on item count
        pq.put(deepcopy(item))
    event.clear()


class _ParallelDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, dataloader, num_workers, batch_size):

        assert num_workers > 0

        super(_ParallelDataLoaderIter, self).__init__(dataloader, batch_size)
        self.num_workers = num_workers
        self.queue = PriorityQueue(maxsize=num_workers * 500)
        # spawn thread
        self.prod_event = Event()
        self.producer = Thread(target=_dataload_thread,
                               args=(self.dataloader, self.queue, self.prod_event))
        self.producer.start()

    def __next__(self):
        items = []
        worker_ids = set()
        block = False
        while len(items) < self.batch_size:
            try:
                item = self.queue.get(block=block, timeout=0.5)
            except Empty:
                if self.prod_event.is_set():
                    # producer still running, items left
                    block = True
                    continue
                else:
                    # producer done, queue empty
                    break
            except TimeoutError:
                continue
            if item[1] in worker_ids:
                self.queue.put(item)
            else:
                worker_ids.add(item[1])
                items.append(item)
                block = False

        if len(items) < self.batch_size:
            # only when preempted from loop
            raise StopIteration
        else:
            return items

    def __del__(self):
        self.producer.join()


class _SeqDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, dataloader, batch_size):
        super(_SeqDataLoaderIter, self).__init__(dataloader, batch_size)
        self.loader_iter = iter(self.dataloader)

    def __next__(self):
        return [next(self.loader_iter) for i in range(self.batch_size)]
