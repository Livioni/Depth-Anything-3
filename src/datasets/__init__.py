
from .scannetppv2 import Scannetppv2 # noqa
from .infinigen import Infinigen # noqa
from .droid import Droid # noqa
from .utils.transforms import ColorJitter, ImgNorm # noqa


def get_data_loader(dataset, batch_size, num_workers=8,
                    shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from src.datasets.utils.misc import get_world_size, get_rank
    
    world_size = get_world_size()
    rank = get_rank()
    if isinstance(dataset, str):
        dataset = eval(dataset)
    
    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
            
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1, #Do not modify this
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=False,
        drop_last=drop_last,
        )   
    
    return data_loader
