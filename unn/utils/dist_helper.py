import os
import torch
import linklink as link

def DistModule(model, sync=True):
    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(name, p, i):
        def hook(*ignore):
            link.allreduce_async(name, p.grad.data)
        return hook

    broadcast_params(model)
    if not sync:
        model._grad_accs = []
        model._register_hooks = _register_hooks
        model._make_hook = _make_hook
        model._register_hooks(model)
    return model


def reduce_gradients(model, sync=True):
    if sync:
        for name, param in model.named_parameters():
            if param.requires_grad:
                link.allreduce(param.grad.data)
    else:
        # reduce all grdients asynchronously, faster
        link.synchronize()


def broadcast_params(model):
    for name, p in model.state_dict().items():
        link.broadcast(p, 0)


def get_rank():
    """Replace linklink.get_rank"""
    return int(os.environ.get('SLURM_PROCID', 0))


def get_world_size():
    """Replace linklink.get_world_size"""
    return int(os.environ.get('SLURM_NTASKS', 1))


def barrier():
    """Replace linklink.barrier"""
    if get_world_size() > 1:
        link.barrier()


def finalize():
    """Relpace linklink.finalize"""
    link.finalize()
    # if get_world_size() > 1:
    #     link.finalize()


def setup_distributed():
    """Do not set device for single gpu"""
    rank = get_rank()
    world_size = get_world_size()

    # IMPORTANT: To support single process without linklink
    if world_size == 1:
        return rank, world_size

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    link.initialize()
    return rank, world_size
