from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from graph_matching.loss import contrastive_loss_cosine


def test_fn(model, test_data_loader, device, temperature=1.0):
    model.eval()
    with torch.no_grad():
        emb_llm_1 = []
        emb_llm_2 = []
        for data in tqdm(test_data_loader, desc="Testing", leave=False):
            batch_emb_llm_1, batch_emb_llm_2 = model.forward_emb(data.to(device))
            emb_llm_1.append(batch_emb_llm_1)
            emb_llm_2.append(batch_emb_llm_2)
        emb_llm_1 = torch.cat(emb_llm_1, dim=0)
        emb_llm_2 = torch.cat(emb_llm_2, dim=0)
        sim_matrix = torch.matmul(emb_llm_1, emb_llm_2.t()) / temperature
        test_loss = contrastive_loss_cosine(sim_matrix).item()
        sim_matrix = sim_matrix.cpu()
        num_tasks = sim_matrix.size(0)
        identity = torch.eye(num_tasks)
        test_auc_1_to_2 = np.array([roc_auc_score(identity[i], sim_matrix[i]) for i in range(num_tasks)]).mean()
        test_auc_2_to_1 = np.array([roc_auc_score(identity[i], sim_matrix[:, i]) for i in range(num_tasks)]).mean()
        test_gauc = (test_auc_1_to_2 + test_auc_2_to_1) / 2
        test_auc = roc_auc_score(identity.flatten(), sim_matrix.flatten())

    return test_loss, test_gauc, test_auc, sim_matrix
