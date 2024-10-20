import torch
from comfy.ldm.modules.attention import optimized_attention

class InstantId(torch.nn.Module):
  def __init__(self, ip_adapter):
    super().__init__()

    self.to_kvs = torch.nn.ModuleDict()

    for key, value in ip_adapter.items():
      k = key.replace(".weight", "").replace(".", "_")
      self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
      self.to_kvs[k].weight.data = value


# based on https://github.com/laksjdjf/IPAdapter-ComfyUI/blob/main/ip_adapter.py#L256
class CrossAttentionPatch:
  def __init__(self, scale, instantId, cond, number):
    self.scales = [scale]
    self.instantIds = [instantId]
    self.conds = [cond]
    self.number = number

  def set_new_condition(self, scale, instantId, cond, number):
    self.scales.append(scale)
    self.instantIds.append(instantId)
    self.conds.append(cond)
    self.number = number

  def __call__(self, q, k, v, extra_options):
    dtype = torch.float16
    hidden_states = optimized_attention(q, k, v, extra_options["n_heads"])
    for scale, cond, instantId in zip(self.scales, self.conds,  self.instantIds):
      k_cond = instantId.to_kvs[str(self.number*2+1) + "_to_k_ip"](cond).to(dtype=dtype)
      v_cond = instantId.to_kvs[str(self.number*2+1) + "_to_v_ip"](cond).to(dtype=dtype)
      ip_hidden_states = optimized_attention(q, k_cond, v_cond, extra_options["n_heads"])
      hidden_states = hidden_states + ip_hidden_states * scale
      return hidden_states.to(dtype=dtype)