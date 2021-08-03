# Main reference: https://pytorch.org/docs/1.5.0/distributed.html
# Stand library
import functools
import os
import pickle
import sys

# import from torch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
import pdb

MASTER_RANK = 0


# def DistModule(model, sync=True):
#     def _register_hooks(self):
#         for i, (name, p) in enumerate(self.named_parameters()):
#             if p.requires_grad:
#                 p_tmp = p.expand_as(p)
#                 grad_acc = p_tmp.grad_fn.next_functions[0][0]
#                 grad_acc.register_hook(self._make_hook(name, p, i))
#                 self._grad_accs.append(grad_acc)

#     def _make_hook(name, p, i):
#         def hook(*ignore):
#             # todo
#             link.allreduce_async(name, p.grad.data)
#             # torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=<object object>, async_op=False)
#             # dist.all_reduce(name, p.grad.data, async_op=True)
#         return hook

#     broadcast_params(model)
#     if not sync:
#         model._grad_accs = []
#         model._register_hooks = _register_hooks
#         model._make_hook = _make_hook
#         model._register_hooks(model)
#     return model


def reduce_gradients(model, sync=True, allow_dead_parameter=False):
    if sync:
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                try:
                    # link.allreduce(param.grad.data)
                    dist.all_reduce(param.grad.data,async_op=False)
                except AttributeError as e:
                    warning = (f"The AttributeError above was probably caused by {name}.grad being None "
                               f"but the gradient w.r.t {name} is required. "
                               f"Please check your model to make sure that {name} is always used in your "
                               "forward pass if it is learnable, otherwise, set it's requrires_grad flag to False. "
                               "Another temporary workaround, you may add 'and param.grad is not None' to "
                               "the conditional above only if you know exactly the grad is needless.")
                    if not allow_dead_parameter:
                        raise AttributeError('This line is not an error, just warning: ' + warning) from e
    else:
        # reduce all grdients asynchronously, faster
        # link.synchronize()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                try:
                    # link.allreduce(param.grad.data)
                    dist.all_reduce(param.grad.data,async_op=True)
                except AttributeError as e:
                    warning = (f"The AttributeError above was probably caused by {name}.grad being None "
                               f"but the gradient w.r.t {name} is required. "
                               f"Please check your model to make sure that {name} is always used in your "
                               "forward pass if it is learnable, otherwise, set it's requrires_grad flag to False. "
                               "Another temporary workaround, you may add 'and param.grad is not None' to "
                               "the conditional above only if you know exactly the grad is needless.")
                    if not allow_dead_parameter:
                        raise AttributeError('This line is not an error, just warning: ' + warning) from e        



def broadcast_params(model):

    for name, p in model.state_dict().items():
        # link.broadcast(p, MASTER_RANK)
        dist.broadcast(p, MASTER_RANK)
        # torch.distributed.broadcast(tensor, src, group=<object object>, async_op=False)
        # Parameters
        # tensor (Tensor) – Data to be sent if src is the rank of current process, and tensor to be used to save received data otherwise.
        # src (int) – Source rank.
        # group (ProcessGroup, optional) – The process group to work on
        # async_op (bool, optional) – Whether this op should be an async op
        # distributed.broadcast(p,MASTER_RANK,async_op=False)


def get_rank_from_env():
    rank_cands = ['SLURM_PROCID', 'MV2_COMM_WORLD_RANK', 'PMI_RANK']
    for rank_name in rank_cands:
        if rank_name in os.environ:
            return int(os.environ[rank_name])
    return None

def get_rank():
    """Replace linklink.get_rank"""
    try:
        rank = get_rank_from_env()
        if rank is not None:
            return rank
        else:
            # return link.get_rank()
            return dist.get_rank()
    except Exception as e:  # noqa
        return 0

def get_local_rank():
    return get_rank() % torch.cuda.device_count()


def get_world_size_from_env():
    ws_cands = ['SLURM_NTASKS', 'MV2_COMM_WORLD_SIZE', 'PMI_SIZE']
    for ws_name in ws_cands:
        if ws_name in os.environ:
            return int(os.environ[ws_name])
    return None

def get_world_size():
    """Replace linklink.get_world_size"""
    try:
        world_size = get_world_size_from_env()
        if world_size is not None:
            return world_size
        else:
            # return link.get_world_size()
            return dist.get_world_size()
    except Exception as e:  # noqa
        return 1
    return dist.get_world_size()


def barrier():
    """Replace linklink.barrier"""
    if get_world_size() > 1:
        # link.barrier()
        dist.barrier()


# def finalize():
#     """Relpace linklink.finalize"""
#     link.finalize()

def setup_env(port):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']

    if "," in node_list:
        node_list = node_list.split(",")[0]

    # extract the master addr
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')

    current_env = os.environ
    current_env['MASTER_PORT'] = str(port)
    current_env['MASTER_ADDR'] = str(addr)
    current_env['WORLD_SIZE'] = str(ntasks)
    current_env['RANK'] = str(proc_id)
    return current_env


def setup_distributed(ddp=False, port=33334):

    if False and ddp:
        setup_env(port)
        rank = get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        # link.init_process_group(backend="nccl", init_method="env://")
        dist.init_process_group(backend="nccl", init_method="env://")
        # link.barrier()
        dist.barrier()
    else:
        if 'SLURM_PROCID' in os.environ:    # slurm mode
            device = get_rank() % torch.cuda.device_count()
            torch.cuda.set_device(device)
            # link.initialize()
        else:
            # link.initialize()
            device = get_rank() % torch.cuda.device_count()
            torch.cuda.set_device(device)
    rank = get_rank()
    world_size = get_world_size()
    print('rank:{} world_size(gpus):{}'.format(rank, world_size))
    sys.stdout.flush()
    return rank, world_size


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    # if link.get_backend() == "nccl":
    if dist.get_backend() == "nccl":
        # torch.distributed.new_group(ranks=None, timeout=datetime.timedelta(0, 1800), backend=None)
        return dist.new_group(backend="gloo")
    else:
        # return link.group.WORLD
        return dist.group.WORLD


## easy communicate picklable object across gpus
def _serialize_to_tensor(data, group=None):
    # backend = link.get_backend(group)
    # assert backend in ["gloo", "nccl"]
    # device = torch.device("cpu" if backend == "gloo" else "cuda")
    device = torch.cuda.current_device()
    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        import logging
        logger = logging.getLogger('global')
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    # world_size = link.get_world_size(group=group)
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    # torch.distributed.all_gather(tensor_list, tensor, group=<object object>, async_op=False)
    # link.all_gather(size_list, local_size, group=group)
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    # if link.get_world_size(group) == 1:
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    # link.all_gather(tensor_list, tensor, group=group)
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=MASTER_RANK, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    # if link.get_world_size(group=group) == 1:
    if dist.get_world_size(group=group) == 1:
        return [data]
    # rank = link.get_rank(group=group)
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
        ]
        # link.gather(tensor, tensor_list, dst=dst, group=group)
        # torch.distributed.gather(tensor, gather_list=None, dst=0, group=<object object>,
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        # link.gather(tensor, [], dst=dst, group=group)
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def broadcast_object(obj, group=None):
    """make suare obj is picklable
    """
    if get_world_size() == 1:
        return obj

    serialized_tensor = _serialize_to_tensor(obj).cuda()
    numel = torch.IntTensor([serialized_tensor.numel()]).cuda()
    # link.broadcast(numel, MASTER_RANK)
    # torch.distributed.broadcast(tensor, src, group=<object object>, async_op=False)
    dist.broadcast(numel, MASTER_RANK)
    # serialized_tensor from storage is not resizable
    serialized_tensor = serialized_tensor.clone()
    serialized_tensor.resize_(numel)
    # link.broadcast(serialized_tensor, MASTER_RANK)
    dist.broadcast(serialized_tensor, MASTER_RANK)
    serialized_bytes = serialized_tensor.cpu().numpy().tobytes()
    deserialized_obj = pickle.loads(serialized_bytes)
    return deserialized_obj


def allreduce(tensor):
    assert torch.is_tensor(tensor)
    if get_world_size() > 1:
        # link.allreduce(tensor)
        dist.all_reduce(tensor)
