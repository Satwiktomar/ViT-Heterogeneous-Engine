import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Standard Linear layer with hooks for Low-Rank Adaptation (LoRA).
    
    Phase 1 (Now): Acts as a standard nn.Linear.
    Phase 2 (Later): We will add 'lora_A' and 'lora_B' parameters.
    """
    def __init__(self, in_features, out_features, bias=True, lora_rank=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Main Weight Matrix (Frozen during fine-tuning later)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

        # --- Future LoRA placeholders ---
        self.lora_rank = lora_rank
        if lora_rank is not None:
            print(f"🔧 Initializing LoRA with rank {lora_rank} (Placeholder)")
            # self.lora_A = nn.Parameter(...)
            # self.lora_B = nn.Parameter(...)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Standard Linear Pass: y = xA^T + b
        return nn.functional.linear(input, self.weight, self.bias)
        
        # Future LoRA Pass:
        # return F.linear(input, self.weight + (self.lora_A @ self.lora_B))