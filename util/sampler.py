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
