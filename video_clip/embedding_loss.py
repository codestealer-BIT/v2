import torch
import torch.nn.functional as F
import numpy as np


class LearnableNTXentLoss(torch.nn.Module):
    def __init__(self, init_tau=0.07, clamp=4.6051):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.tensor([np.log(1.0 / init_tau)], dtype=torch.float32))
        self.calc_ce = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.clamp = clamp  # 4.6051 等价于CLAMP 100, 初始值是2.6593，

    def forward(self, v_emb=None, t_emb=None, logits=None, valid_idx=None, use_double_loss=True):
        """
        v_emb: batch 对比loss的一边
        t_emb: batch 对比loss的另一边
        logits: 需要计算对比loss的矩阵，default: None
        """
        self.tau.data = torch.clamp(self.tau.data, 0, self.clamp)
        if logits is None:
            bsz = v_emb.shape[0]
            v_emb = F.normalize(v_emb, dim=1)
            t_emb = F.normalize(t_emb, dim=1)
            logits = torch.mm(v_emb, t_emb.t()) * self.tau.exp()  # [bsz, bsz]
        else:
            bsz = logits.shape[0]
            logits = logits * self.tau.exp()
        
        labels = torch.arange(bsz, device=logits.device)  # bsz
        if valid_idx is None:
            loss_v = self.calc_ce(logits, labels).mean()
            if use_double_loss:
                loss_t = self.calc_ce(logits.t(), labels).mean()
            else:
                loss_t = 0.0
        else:
            loss_v = self.calc_ce(logits, labels)[:valid_idx].mean()
            if use_double_loss:
                loss_t = self.calc_ce(logits.t(), labels)[:valid_idx].mean()
            else:
                loss_t = 0.0
        
        if use_double_loss:
            loss = (loss_v + loss_t) / 2
        else:
            loss = loss_v

        return loss