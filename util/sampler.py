import torch


class RandomSubsetSampler(torch.utils.data.Sampler):
    """
    Draws a fresh random subset of the dataset every epoch (without replacement), split across ranks.
    Lets an "epoch" be a fixed number of steps on very large datasets. Call set_epoch() each epoch,
    as with DistributedSampler, so every epoch (and every resume) sees a different subset.
    """

    def __init__(self, dataset, num_samples, num_replicas=1, rank=0, seed=0):
        self.dataset_size = len(dataset)
        self.num_samples = num_samples  # per rank
        self.total_size = num_samples * num_replicas
        assert self.total_size <= self.dataset_size, \
            "steps_per_epoch * batch_size * world_size ({}) exceeds dataset size ({})".format(self.total_size, self.dataset_size)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # same permutation on every rank, each rank takes a disjoint slice
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=g)[:self.total_size]
        return iter(indices[self.rank::self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples


class WidthBinBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler for panoramas of different widths: every batch only holds samples of one width (one bin),
    so they can be stacked. widths: one width per dataset index (dataset.widths).
    Every epoch, the indices of each bin are shuffled and cut into full batches, and the batches of all bins are
    shuffled together, so a width is drawn in proportion to its number of samples. Batches are dealt to the ranks
    in turn; with num_batches (per rank) an epoch is a fresh random subset of that many batches, as with
    RandomSubsetSampler. Call set_epoch() each epoch.
    """

    def __init__(self, widths, batch_size, num_batches=None, num_replicas=1, rank=0, seed=0):
        import numpy as np
        widths = np.asarray(widths)
        order = np.argsort(widths, kind='stable')
        values, starts = np.unique(widths[order], return_index=True)
        self.bins = dict(zip(values.tolist(), np.split(order, starts[1:])))
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        total = sum(len(b) // batch_size for b in self.bins.values()) // num_replicas
        if num_batches is not None:
            assert num_batches <= total, \
                "steps_per_epoch ({}) exceeds the full batches per rank available ({})".format(num_batches, total)
        self.num_batches = num_batches if num_batches is not None else total

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        import numpy as np
        # same batches on every rank, each rank takes every num_replicas-th batch
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for indices in self.bins.values():
            indices = rng.permutation(indices)
            n = len(indices) // self.batch_size * self.batch_size
            batches += np.split(indices[:n], n // self.batch_size) if n else []
        order = rng.permutation(len(batches))
        mine = order[self.rank::self.num_replicas][:self.num_batches]
        return iter([batches[i].tolist() for i in mine])

    def __len__(self):
        return self.num_batches

    def __repr__(self):
        return "WidthBinBatchSampler({} widths {}..{}, batch_size={}, {} batches per epoch)".format(
            len(self.bins), min(self.bins), max(self.bins), self.batch_size, self.num_batches)
