import os
import torch.distributed as dist
import torch
from torch.multiprocessing import Process
from dataclasses import dataclass
from torch import nn
from torch.nn.functional import gelu
from typing import Literal
from rich.pretty import pprint
from copy import deepcopy

@dataclass
class ModelConfig:
    d_model: int = 128
    d_hidden: int = 128
    act: Literal["swiglu", "gelu"] = "gelu"


def swiglu(x):
    act, mul = x.chunk(2, dim=-1)
    return gelu(act) * mul


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
        x = self.in_proj(x)
        x = self.act_fn(x)
        x = self.out_proj(x)
        if self.is_tp_split:
            dist.all_reduce(x, group=tp_group)
        return x

    def make_tp(self, rank: int, world_size: int) -> "Mlp":
        ret = Mlp(ModelConfig(self.conf.d_model, self.conf.d_hidden // world_size, self.conf.act))
        split_sd = {
            "in_proj.weight": self.in_proj.weight.chunk(world_size, dim=0)[rank],
            "out_proj.weight": self.out_proj.weight.chunk(world_size, dim=1)[rank],
        }
        ret.load_state_dict(split_sd)
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
    norm = torch.cat([x.grad.flatten() for x in model.parameters()]).norm()  # type:ignore
    dist.all_reduce(norm, group=reduce_group)
    return norm


def test(info: ProcessInfo, conf: ModelConfig, tp_group, dp_group):
    model = Model(conf)
    tp_model = model.make_tp(info.tp_rank, info.tp_world_size)

    x = torch.randn((info.dp_world_size, conf.d_model))[info.dp_rank]
    dy = torch.randn((info.dp_world_size, conf.d_model))[info.dp_rank]
    baseline = model(x, tp_group)
    tp = tp_model(x, tp_group)
    baseline.backward(dy)
    tp.backward(dy)

    baseline_gradnorm = grad_norm(model, dp_group)
    tp_gradnorm = grad_norm(tp_model, None)

    if info.tp_rank == 0 and info.dp_rank == 0:
        print("")
        pprint(conf)
        print_relative_error(baseline, tp)
        print_relative_error(baseline_gradnorm, tp_gradnorm)


def entry_point(info: ProcessInfo):
    torch.manual_seed(42)
    tp_group, dp_group = init_dist(info)
    test(info, ModelConfig(act="gelu"), tp_group, dp_group)
    test(info, ModelConfig(act="swiglu"), tp_group, dp_group)
    dist.destroy_process_group()


if __name__ == "__main__":

    tp_world_size = 2
    dp_world_size = 2

    processes: list[Process] = []
    for tp_rank in range(tp_world_size):
        for dp_rank in range(dp_world_size):
            p = Process(target=entry_point, args=(ProcessInfo(tp_rank, dp_rank, tp_world_size, dp_world_size),))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
