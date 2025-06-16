import torch
import torch.nn.functional as F


def contrastive_loss(z, mask):
    z = F.normalize(z, dim=1, p=2)
    s = torch.mm(z, z.t())
    pos_mask = mask.fill_diagonal_(0)
    neg_mask = 1 - mask.fill_diagonal_(1)

    tau = 1.0 # 1.0
    s = torch.exp(s / tau)
    pos_loss = (pos_mask * s).sum(1)
    neg_loss = (neg_mask * s).sum(1)
    loss_con = - torch.log(pos_loss / neg_loss).mean()

    return loss_con


def contrastive_loss2(z, mask):
    z = F.normalize(z, dim=1, p=2)
    s = torch.mm(z, z.t())
    pos_mask = mask.fill_diagonal_(0)
    neg_mask = 1 - mask.fill_diagonal_(1)

    tau = 1.0 # 1.0
    s = torch.exp(s / tau)
    pos_loss = (pos_mask * s).sum(1)
    neg_loss = (neg_mask * s).sum(1)
    loss_con = - torch.log(pos_loss / (pos_loss + neg_loss)).mean()

    return loss_con

def knn_graph(distances, k=15):
    N = distances.shape[0]
    idx = torch.argsort(distances, dim=1)[:, :k + 1]  # Shape (N, k + 1)
    neighbors_idx = idx[:, 1:k + 1]  # Exclude the first column (self)
    d = distances[torch.arange(N).unsqueeze(1), neighbors_idx]  # Shape (N, k)
    adjacency_matrix = torch.zeros((N, N), dtype=distances.dtype, device=distances.device)
    eps = 1e-8
    d_k_minus_1 = d[:, -1]  # k-th nearest distance (last in the row)
    sum_d = torch.sum(d, dim=1)  # Sum of distances for each row
    adjacency_matrix[torch.arange(N).unsqueeze(1), neighbors_idx] \
        = (d_k_minus_1.unsqueeze(1) - d) / (k * d_k_minus_1.unsqueeze(1) - sum_d.unsqueeze(1) + eps)
    adjacency_matrix.fill_diagonal_(1)
    adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.t())
    return adjacency_matrix

