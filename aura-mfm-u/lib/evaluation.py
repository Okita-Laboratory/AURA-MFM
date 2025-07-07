# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import json
from tqdm.auto import tqdm
from torchmetrics import RetrievalRecall, RetrievalMRR


def compute_metrics(source_embeddings, target_embeddings):
    """
    input:
    - source_embeddings: (n, m)
    - target_embeddings: (n, m)
    output:
    - Recall@1
    - Recall@10
    - Recall@50
    - MRR
    """
    # prepare metrics
    compute_mrr = RetrievalMRR()
    compute_r1 = RetrievalRecall(k=1)
    compute_r10 = RetrievalRecall(k=10)
    compute_r50 = RetrievalRecall(k=50)
    s_t_metrics = {"MRR": 0, "R@1": 0, "R@10": 0, "R@50": 0}
    t_s_metrics = {"MRR": 0, "R@1": 0, "R@10": 0, "R@50": 0}
    n = source_embeddings.shape[0]
    # print(f"the number of queries & candidates = {n}")
    target = torch.eye(n).view(-1)
    indexes = torch.arange(n).repeat(n, 1).transpose(0, 1)
    indexes = indexes.reshape(-1)
    #  Compute similarity
    s = torch.nn.functional.normalize(source_embeddings, dim=1)
    t = torch.nn.functional.normalize(target_embeddings, dim=1)
    tt = t.transpose(0, 1)
    st = s.transpose(0, 1)
    # Do query batch by batch to avoid OOM issue.
    bsz = 1024
    batch_num = n // bsz +1
    # print("Start batch retrieval:")
    # s -> t
    s_t_batch_results = []
    for i in range(batch_num):
        start = i * bsz
        end = min((i + 1) * bsz, n)
        query_batch = torch.mm(s[start:end], tt)  # (bsz, m) (m, n) -> (bsz, n)
        s_t_batch_results.append(query_batch.detach().cpu())
    s_t_batch_results = torch.cat(s_t_batch_results, dim=0).view(-1)  # (n,n)

    mrr = compute_mrr(s_t_batch_results, target, indexes=indexes).item()
    r1 = compute_r1(s_t_batch_results, target, indexes=indexes).item()
    r10 = compute_r10(s_t_batch_results, target, indexes=indexes).item()
    r50 = compute_r50(s_t_batch_results, target, indexes=indexes).item()
    s_t_metrics = {"MRR": mrr, "R@1": r1, "R@10": r10, "R@50": r50}

    # t -> s
    t_s_batch_results = []
    for i in range(batch_num):
        start = i * bsz
        end = min((i + 1) * bsz, n)
        query_batch = torch.mm(t[start:end], st)  # (bsz, m) (m, n) -> (bsz, n)
        t_s_batch_results.append(query_batch.detach().cpu())
    t_s_batch_results = torch.cat(t_s_batch_results, dim=0).view(-1)  # (n,n)

    mrr = compute_mrr(t_s_batch_results, target, indexes=indexes).item()
    r1 = compute_r1(t_s_batch_results, target, indexes=indexes).item()
    r10 = compute_r10(t_s_batch_results, target, indexes=indexes).item()
    r50 = compute_r50(t_s_batch_results, target, indexes=indexes).item()
    t_s_metrics = {"MRR": mrr, "R@1": r1, "R@10": r10, "R@50": r50}

    return s_t_metrics, t_s_metrics


def evaluate(test_loader, model, source='imu', target='mocap', result_path=None, device='cpu', configs=None):
    model.to(device)
    model.eval()
    queries, keys = [], []
    for x in test_loader:
        with torch.no_grad():
            out = model(x)
            queries.append(out[source])
            keys.append(out[target])
    
    y_query_modality = torch.cat(queries, dim=0)
    y_key_modality   = torch.cat(keys, dim=0)

    s_t_metrics, t_s_metrics = compute_metrics(y_query_modality, y_key_modality)

    # Save metrics
    num_candidates = y_query_modality.shape[0]
    metrics = {
        "s_t_metrics": s_t_metrics,
        "t_s_metrics": t_s_metrics,
        "num_candidates": num_candidates,
    }
    if result_path is not None:
        result_path = os.path.join(result_path, f'candi_num_{num_candidates}.json')
        with open(result_path, "w") as f:
            json.dump({"metrics": metrics, "configs": configs}, f, indent=4)
    return metrics



def evaluate_model(model, loader, loss_fn, device='cpu'):
    model.eval()
    running_loss = 0.0
    for x in loader:
        with torch.no_grad():
            out = model(x)
            loss = loss_fn(out)
            running_loss += loss.item()
    return running_loss/len(loader)