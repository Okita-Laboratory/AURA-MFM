# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import json

import torch
from torchmetrics import RetrievalMRR, RetrievalRecall
from tqdm import tqdm
import logging
import numpy as np
import sys
import os

def evaluate(
    test_set,
    collate_fn,
    model,
    source_modality,
    target_modalities,
    result_path,
    configs=None,
):
    dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=configs["train_hparams"]["batch_size"],  # 32,
        num_workers=configs["train_hparams"]["num_workers_for_dm"],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    # np.save("video_989",test_set[989]["video"]["frames"].numpy())
    # np.save("video_1010",test_set[1010]["video"]["frames"].numpy())
    # # np.save("video_500",test_set[]["video"]["frames"].numpy())
    # sys.exit()
    # np.save("video_0_fromdataset",test_set[0]["video"]["frames"].numpy())
    logging.info(f"len_dataloader:{len(dataloader)}")
    device = torch.device("cuda:0")
    target_modalities.sort()
    list_modalities = [source_modality] + target_modalities
    if "imu" in list_modalities:
        imu_encoder = model.imu_encoder
        imu_encoder.to(device)
        imu_encoder.eval()
    if "text" in list_modalities:
        text_encoder = model.text_encoder
        text_encoder.to(device)
        text_encoder.eval()
    if "video" in list_modalities:
        video_encoder = model.video_encoder
        video_encoder.to(device)
        video_encoder.eval()
    if "audio" in list_modalities:
        audio_encoder = model.audio_encoder
        audio_encoder.to(device)
        audio_encoder.eval()
        
    count = 0
    out = {"imu": [], "text": [], "video": [], "audio": []}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if "imu" in list_modalities:
                # if count == 0:
                #     np.save("imu_0",batch["imu"][0].numpy())
                #     np.save("video_0",batch["video"][0].numpy())
                #     count += 1
                x_imu = batch["imu"].to(device)
                # print(f"x_imu_shape:{x_imu.shape}")
                y_imu = imu_encoder(x_imu)
                out["imu"].append(y_imu.cpu())

            if "text" in list_modalities:
                x_narration = batch["narration"]
                # print(f"x_narration:{x_narration}")
                y_narration = text_encoder.get_text_embeddings(x_narration, device)
                # print(f"y_narration:{y_narration}")
                out["text"].append(y_narration.cpu())

            if "video" in list_modalities:
                x_video = batch["video"].to(device)
                # print(f"x_video_shape:{x_video.shape}")
                y_video = video_encoder.get_video_embeddings(x_video)
                # print(f"y_video:{y_video}")
                out["video"].append(y_video.cpu())

            if "audio" in list_modalities:
                x_audio = batch["audio"].to(device)
                y_audio = audio_encoder.get_audio_embeddings(x_audio)
                out["audio"].append(y_audio.cpu())

    if not configs["train_hparams"]["source_index"] == -1:
        source_index = configs["train_hparams"]["source_index"]
    else:
        source_index = -1
    if source_index == -1:
        y_query_modality = torch.cat(out[source_modality], dim=0)
    else:
        y_query_modality = torch.unsqueeze(torch.cat(out[source_modality], dim=0)[source_index],0)
    # text_encoder = model.text_encoder
    # text_encoder.to(device)
    # text_encoder.eval()
    # y_query_modality = text_encoder.get_text_embeddings(['A woman is bouldering indoors, climbing a colorful artificial rock wall. She wears a white tank top, striped leggings, and climbing shoes, using chalk for grip while reaching for a hold.'], device).cpu()
    print(f"y_query_modality_shape:{y_query_modality.shape}")

    if "text" in target_modalities:
        y_key_modality = torch.cat(out["text"], dim=0)

    elif "video" in target_modalities:
        y_key_modality = torch.cat(out["video"], dim=0)

    elif "audio" in target_modalities:
        y_key_modality = torch.cat(out["audio"], dim=0)

    s_t_metrics, t_s_metrics = compute_metrics(y_query_modality, y_key_modality, source_index=source_index)

    # Save metrics
    num_candidates = y_query_modality.shape[0]
    metrics = {
        "s_t_metrics": s_t_metrics,
        "t_s_metrics": t_s_metrics,
        "num_candidates": num_candidates,
    }
    result_path += f"_candi_num_{num_candidates}.json"
    os.makedirs(os.path.dirname(os.path.normpath(result_path)), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({"metrics": metrics, "configs": configs}, f, indent=4)

    return metrics


def compute_metrics(source_embeddings, target_embeddings, source_index=None):
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
    source_embeddings.cpu()
    target_embeddings.cpu()
    compute_mrr = RetrievalMRR()
    compute_r1 = RetrievalRecall(top_k=1)
    compute_r10 = RetrievalRecall(top_k=10)
    compute_r50 = RetrievalRecall(top_k=50)
    s_t_metrics = {"MRR": 0, "R@1": 0, "R@10": 0, "R@50": 0}
    t_s_metrics = {"MRR": 0, "R@1": 0, "R@10": 0, "R@50": 0}
    n = source_embeddings.shape[0]
    logging.info(f"the number of queries & candidates = {n}")
    # target = torch.eye(n).cuda(0).view(-1)
    if source_index == -1:
        target = torch.eye(n).view(-1)
        indexes = torch.arange(n).repeat(n, 1).transpose(0, 1)
        indexes = indexes.reshape(-1)
    else:
        target = torch.zeros(target_embeddings.shape[0])
        target[source_index] = 1
        indexes = torch.zeros(target_embeddings.shape[0],dtype=torch.long)
    #  Compute similarity
    s = torch.nn.functional.normalize(source_embeddings, dim=1)
    t = torch.nn.functional.normalize(target_embeddings, dim=1)
    tt = t.transpose(0, 1)
    st = s.transpose(0, 1)
    # Do query batch by batch to avoid OOM issue.
    bsz = 1  # 32
    logging.info(f"n_shape:{n}")
    batch_num = n // bsz
    logging.info("Start batch retrieval:")
    # s -> t
    s_t_batch_results = []
    for i in tqdm(range(batch_num)):
        start = i * bsz
        end = min((i + 1) * bsz, n)
        query_batch = torch.mm(s[start:end], tt)  # (bsz, m) (m, n) -> (bsz, n)
        s_t_batch_results.append(query_batch)
    s_t_batch_results = torch.cat(s_t_batch_results, dim=0).view(-1)  # (n,n)
    # 最大値のインデックスを取得
    print(f"s_t_batch_results_shape:{s_t_batch_results.shape}")
    first_index = torch.argsort(s_t_batch_results)[-1]
    first_index = torch.argmax(s_t_batch_results)
    second_index = torch.argsort(s_t_batch_results)[-2]
    third_index = torch.argsort(s_t_batch_results)[-3]
    fifty_index = torch.argsort(s_t_batch_results)[-50]
    # 最大値を取得
    first_value = s_t_batch_results[first_index]
    second_value = s_t_batch_results[second_index]
    third_value = s_t_batch_results[third_index]
    fifty_value = s_t_batch_results[fifty_index]

    logging.info(f"first_index:{first_index},first_value:{first_value}, second_index:{second_index},second_value:{second_value}, third_index:{third_index},third_value:{third_value}, fifty_index:{fifty_index},fifty_value:{fifty_value}")
    # s_t_batch_results.to(target.device)
    #logging.info(f"target.device:{target.device},s_t_batch_results.device:{s_t_batch_results.device}")
    # 最大値のインデックスを取得
    # max_index = np.argmax(s_t_batch_results[:1280])
    # print(f"max_index:{max_index}")
    # print(f"s_t_batch_results_shape:{s_t_batch_results.shape}")
    # # 最大値を取得
    # max_value = s_t_batch_results[max_index]

    # print(f"最大値: {max_value}, インデックス: {max_index}")
    # # s_t_batch_results.to(target.device)
    # logging.info(f"target.device:{target.device},s_t_batch_results.device:{s_t_batch_results.device}")
    mrr = compute_mrr(s_t_batch_results, target, indexes=indexes).item()
    r1 = compute_r1(s_t_batch_results, target, indexes=indexes).item()
    r10 = compute_r10(s_t_batch_results, target, indexes=indexes).item()
    r50 = compute_r50(s_t_batch_results, target, indexes=indexes).item()
    s_t_metrics = {"MRR": mrr, "R@1": r1, "R@10": r10, "R@50": r50}

    # t -> s
    if source_index == -1:
        t_s_batch_results = []
        for i in tqdm(range(batch_num)):
            start = i * bsz
            end = min((i + 1) * bsz, n)
            query_batch = torch.mm(t[start:end], st)  # (bsz, m) (m, n) -> (bsz, n)
            t_s_batch_results.append(query_batch)
        t_s_batch_results = torch.cat(t_s_batch_results, dim=0).view(-1)  # (n,n)
        mrr = compute_mrr(t_s_batch_results, target, indexes=indexes).item()
        r1 = compute_r1(t_s_batch_results, target, indexes=indexes).item()
        r10 = compute_r10(t_s_batch_results, target, indexes=indexes).item()
        r50 = compute_r50(t_s_batch_results, target, indexes=indexes).item()
        t_s_metrics = {"MRR": mrr, "R@1": r1, "R@10": r10, "R@50": r50}
    else:
        t_s_metrics = s_t_metrics

    return s_t_metrics, t_s_metrics
