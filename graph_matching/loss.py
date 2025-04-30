import torch
import torch.nn.functional as F


def contrastive_loss_cosine(sim_matrix):
    batch_size = sim_matrix.size(0)
    labels = torch.arange(batch_size).to(sim_matrix.device)

    loss_llm_to_human = F.cross_entropy(sim_matrix, labels)
    loss_human_to_llm = F.cross_entropy(sim_matrix.t(), labels)
    loss = (loss_llm_to_human + loss_human_to_llm) / 2
    return loss
