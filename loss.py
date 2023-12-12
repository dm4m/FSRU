"""
An Lao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-5
T1 = 0.05  # 0.05
T2 = 0.05  # 0.05

class SelfContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=T1):
        super(SelfContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size, batch_size, dtype=torch.bool).to(device)).float())

    def forward(self, q, k):
        q = F.normalize(q, dim=1)  # (bs, dim)  --->  (bs, dim)
        k = F.normalize(k, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([q, k], dim=0)  # (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0), dim=2)  # (2*bs, 2*bs)
        sim_qk = torch.diag(similarity_matrix, self.batch_size)  # (bs,)
        sim_kq = torch.diag(similarity_matrix, -self.batch_size)  # (bs,)

        nominator_qk = torch.exp(sim_qk / self.temperature)   # (bs,)
        negatives_qk = similarity_matrix[:self.batch_size, self.batch_size:]  # (bs, bs)
        denominator_qk = nominator_qk + torch.sum(self.negatives_mask * torch.exp(negatives_qk/self.temperature), dim=1)

        nominator_kq = torch.exp(sim_kq / self.temperature)
        negatives_kq = similarity_matrix[self.batch_size:, :self.batch_size]
        denominator_kq = nominator_kq + torch.sum(self.negatives_mask * torch.exp(negatives_kq/self.temperature), dim=1)

        loss_qk = torch.sum(-torch.log(nominator_qk / denominator_qk + eps)) / self.batch_size
        loss_kq = torch.sum(-torch.log(nominator_kq / denominator_kq + eps)) / self.batch_size
        loss = loss_qk + loss_kq

        return loss

class FullContrastiveLoss(nn.Module):
    def __init__(self, batch_size, num_r, num_nr, device='cuda', temperature=T2):
        super(FullContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.num_r = num_r
        self.num_nr = num_nr

        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("rumor_mask", (~torch.eye(num_r, num_r, dtype=torch.bool).to(device)).float())
        self.register_buffer("nonrumor_mask", (~torch.eye(num_nr, num_nr, dtype=torch.bool).to(device)).float())

    def compute_loss(self, feature, label):
        """
        feature: (batch, dim)
        r: rumor nr: non-rumor
        """
        index_r = torch.nonzero(label).squeeze()
        index_nr = torch.nonzero(label == 0).squeeze()
        ft_r = torch.index_select(feature, dim=0, index=index_r)
        ft_nr = torch.index_select(feature, dim=0, index=index_nr)

        similarity_matrix_r = F.cosine_similarity(ft_r.unsqueeze(1), ft_r.unsqueeze(0), dim=2)
        similarity_matrix_nr = F.cosine_similarity(ft_nr.unsqueeze(1), ft_nr.unsqueeze(0), dim=2)
        similarity_matrix_r_nr = F.cosine_similarity(ft_r.unsqueeze(1), ft_nr.unsqueeze(0), dim=2)
        similarity_matrix_nr_r = F.cosine_similarity(ft_nr.unsqueeze(1), ft_r.unsqueeze(0), dim=2)

        nominator_r = torch.sum(self.rumor_mask * torch.exp(similarity_matrix_r / self.temperature), dim=1)
        nominator_nr = torch.sum(self.nonrumor_mask * torch.exp(similarity_matrix_nr / self.temperature), dim=1)

        denominator_r = nominator_r + torch.sum(torch.torch.exp(similarity_matrix_r_nr / self.temperature), dim=1)
        denominator_nr = nominator_nr + torch.sum(torch.torch.exp(similarity_matrix_nr_r / self.temperature), dim=1)

        loss_r = torch.sum(-torch.log(nominator_r / denominator_r + eps)) / self.num_r
        loss_nr = torch.sum(-torch.log(nominator_nr / denominator_nr + eps)) / self.num_nr
        loss = loss_r + loss_nr
        return loss

    def forward(self, text, image, label):
        text = F.normalize(text, dim=1)
        image = F.normalize(image, dim=1)

        loss = self.compute_loss(text, label) + self.compute_loss(image, label)

        return loss
