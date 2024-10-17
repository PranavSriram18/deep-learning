import os
import torch.distributed as dist
import torch
from torch.multiprocessing import Process
from dataclasses import dataclass
from torch import nn
from torch.nn.functional import gelu
from typing import Literal
#from rich.pretty import pprint
from copy import deepcopy

@dataclass
class ModelConfig:
    d_model: int = 128
    d_hidden: int = 128
    act: Literal["swiglu", "gelu"] = "gelu"


def swiglu(x):
    act, mul = x.chunk(2, dim=-1)
    return gelu(act) * mul

# Fix: Add custom op for reduce
class AllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group):
        """Reduce input across processes in group."""
        output = input.clone()
        dist.all_reduce(output, group=group)
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    

def all_reduce(input, group):
    return AllReduceFunction.apply(input, group)

# Fix: add custom op for broadcast
class BroadcastFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group):
        """Broadcast input across processes in group."""
        ctx.group = group
        output = input.clone()
        # Use the first rank in the group as the source
        src_rank = dist.get_process_group_ranks(group)[0]
        dist.broadcast(output, src=src_rank, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Reduce gradient across processes in group."""
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None
    

def broadcast(input, group):
    return BroadcastFunction.apply(input, group)

class Mlp(nn.Module):
    def __init__(self, conf: ModelConfig):
        super().__init__()

        if conf.act == "swiglu":
            self.in_proj = nn.Linear(conf.d_model, conf.d_hidden * 2, bias=False)
        else:
            self.in_proj = nn.Linear(conf.d_model, conf.d_hidden, bias=False)

        self.out_proj = nn.Linear(conf.d_hidden, conf.d_model, bias=False)
        self.act_fn = gelu if conf.act == "gelu" else swiglu
        self.conf = conf
        self.is_tp_split = False

    def forward(self, x, tp_group):
        # fix: use custom ops for broadcast and reduce
        if self.is_tp_split:
            x = broadcast(x, tp_group)
        x = self.in_proj(x)
        x = self.act_fn(x)
        x = self.out_proj(x)
        if self.is_tp_split:
            x = all_reduce(x, tp_group)
        return x

    def make_tp(self, rank: int, world_size: int) -> "Mlp":
        # Fix: handle swiglu sharding properly
        new_hidden_size = self.conf.d_hidden // world_size
        ret = Mlp(ModelConfig(self.conf.d_model, new_hidden_size, self.conf.act))
        
        if self.conf.act == "swiglu":
            # For SwiGLU, we need to shard both halves of the hidden state
            weight_chunks = self.in_proj.weight.chunk(world_size * 2, dim=0)
            in_proj_weight = torch.cat([
                weight_chunks[rank],
                weight_chunks[rank + world_size]
            ])
        else:
            # For GELU, we can shard normally
            in_proj_weight = self.in_proj.weight.chunk(world_size, dim=0)[rank]
        
        out_proj_weight = self.out_proj.weight.chunk(world_size, dim=1)[rank]
        
        new_state_dict = {
            "in_proj.weight": in_proj_weight,
            "out_proj.weight": out_proj_weight
        }
        
        ret.load_state_dict(new_state_dict, strict=False)
        ret.is_tp_split = True
        return ret


class Model(nn.Module):
    def __init__(self, conf: ModelConfig):
        super().__init__()
        self.conf = conf
        self.layer_0 = Mlp(conf)
        self.layer_1 = Mlp(conf)

    def forward(self, x, tp_group):
        x = self.layer_0(x, tp_group)
        x = self.layer_1(x, tp_group)
        return x

    def make_tp(self, rank: int, world_size: int) -> "Model":
        ret = deepcopy(self)
        ret.layer_0 = ret.layer_0.make_tp(rank, world_size)
        ret.layer_1 = ret.layer_1.make_tp(rank, world_size)
        return ret

@dataclass
class ProcessInfo:
    tp_rank: int
    dp_rank: int
    tp_world_size: int
    dp_world_size: int

    master_address: str = "localhost"
    master_port: int = 7777

    @property
    def world_size(self):
        return self.dp_world_size * self.tp_world_size

    @property
    def rank(self):
        return self.tp_world_size * self.dp_rank + self.tp_rank


def init_dist(info: ProcessInfo):
    os.environ["MASTER_ADDR"] = info.master_address
    os.environ["MASTER_PORT"] = str(info.master_port)
    os.environ["RANK"] = str(info.rank)
    os.environ["WORLD_SIZE"] = str(info.world_size)
    # os.environ["GLOO_SOCKET_IFNAME"] = "en0"
    dist.init_process_group("gloo", init_method="env://")
    dist.barrier()

    tp_groups_ranks = [
        list(range(dp_rank * info.tp_world_size, ((dp_rank + 1) * info.tp_world_size)))
        for dp_rank in range(info.dp_world_size)
    ]

    dp_groups_ranks = [
        list(range(tp_rank, info.world_size, info.tp_world_size)) for tp_rank in range(info.tp_world_size)
    ]

    assert (
        info.rank in tp_groups_ranks[info.dp_rank]
    ), f"process not in its own tp_group {info.rank}, {tp_groups_ranks[info.dp_rank]}"
    assert (
        info.rank in dp_groups_ranks[info.tp_rank]
    ), f"process not in its own dp_group {info.rank}, {dp_groups_ranks[info.tp_rank]}"

    tp_group = [dist.new_group(group) for group in tp_groups_ranks][info.dp_rank]
    dp_group = [dist.new_group(group) for group in dp_groups_ranks][info.tp_rank]

    return tp_group, dp_group


def relative_error(a, b):
    return (a - b).abs().mean() / (b.abs().mean())


def print_relative_error(a, b):
    err = relative_error(a, b)
    if err < 1e-5:
        print(f"✅ {err=}")
    else:
        print(f"❌ {err=}")


def grad_norm(model: nn.Module, reduce_group):
    with torch.no_grad():            
        # Fix: compute local squared L2 norm, then sqrt only after reduce
        local_norm_sq = sum(w.grad.flatten().dot(w.grad.flatten()) for w in model.parameters() if w.grad is not None)
        local_norm_sq = local_norm_sq.clone().detach()
        dist.all_reduce(local_norm_sq, group=reduce_group)
        global_norm = torch.sqrt(local_norm_sq)

    return global_norm

# Minor fix: pass data as arguments to test
def test(
        info: ProcessInfo, 
        conf: ModelConfig, 
        tp_group, 
        dp_group,
        x, 
        dy
    ):
    model = Model(conf)
    tp_model = model.make_tp(info.tp_rank, info.tp_world_size)

    x_slice = x[info.dp_rank]
    dy_slice = dy[info.dp_rank]

    baseline = model(x_slice, tp_group)
    tp = tp_model(x_slice, tp_group)
    baseline.backward(dy_slice)
    tp.backward(dy_slice)

    baseline_gradnorm = grad_norm(model, dp_group)
    tp_gradnorm = grad_norm(tp_model, None)

    if info.tp_rank == 0 and info.dp_rank == 0:
        print("")
        print(conf)
        print_relative_error(baseline, tp)
        print_relative_error(baseline_gradnorm, tp_gradnorm)


def entry_point(info: ProcessInfo, x, dy):
    torch.manual_seed(42)
    tp_group, dp_group = init_dist(info)
    test(info, ModelConfig(act="gelu"), tp_group, dp_group, x, dy)
    test(info, ModelConfig(act="swiglu"), tp_group, dp_group, x, dy)
    dist.destroy_process_group()

def create_data(dp_world_size, d_model):
    torch.manual_seed(42)
    x = torch.randn((dp_world_size, d_model))
    dy = torch.randn((dp_world_size, d_model))
    return x, dy

if __name__ == "__main__":

    tp_world_size = 2
    dp_world_size = 2

    # Minor fix: generate data upfront instead of in each process, to ensure
    # consistency across processes in a tp group
    x, dy = create_data(dp_world_size, d_model=ModelConfig().d_model)

    processes: list[Process] = []
    for tp_rank in range(tp_world_size):
        for dp_rank in range(dp_world_size):
            p = Process(
                target=entry_point, 
                args=(ProcessInfo(tp_rank, dp_rank, tp_world_size, dp_world_size), x, dy))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
