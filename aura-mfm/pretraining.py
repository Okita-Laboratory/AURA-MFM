# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
import os
import random
from argparse import ArgumentParser
import datetime

import pytorch_lightning as pl
import torch
import yaml
from dataset.ego4d.dataloader import clean_narration_text, filter_narration
from lib.clip_model import ClipPLModel
from lib.data_modules import Ego4dDataModule, Split, UnsupEgo4dDataModule
from lib.evaluation import evaluate
from lib.imu_models import MW2StackRNNPooling, PatchTransformer, AttentionPooledIMUEncoder
from lib.train_modules import MultimodalContrastiveLearningModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import os 
import codecs
import sys
import logging
import lib.transformer as Ts


def setup_logger(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "out.log")
    
    # ロガー設定
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 標準出力もログに出力
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

def train(configs):

    random.seed(1234)

    # Load Model Parameters
    model_hparams = configs.get("model_hparams", {})
    model_name = model_hparams.get("model_name")
    model_suffix = model_hparams.get("model_suffix", "")
    imu_encoder_name = model_hparams.get("imu_encoder_name")
    audio_encoder_name = model_hparams.get("audio_encoder_name")
    video_encoder_name = model_hparams.get("video_encoder_name")
    text_encoder_name = model_hparams.get("text_encoder_name")
    window_sec = model_hparams.get("window_sec")
    target_fps = model_hparams.get("target_fps")
    datasetname = model_hparams.get("datasetname", "ego4d")
    imu_sampling_rate = model_hparams.get("imu_sampling_rate", 200 if datasetname == "ego4d" else 1000)
    final_embedding_size = model_hparams.get("final_embedding_size", 512)

    # Params for the trainer
    train_hparams = configs.get("train_hparams", {})
    source_modality = train_hparams.get("source_modality")
    target_modalities = train_hparams.get("target_modalities")
    limit_train_batches = train_hparams.get("limit_train_batches")
    limit_val_batches = train_hparams.get("limit_val_batches")
    # limit_test_batches = train_hparams.get("limit_test_batches")
    batch_size = train_hparams.get("batch_size")
    max_epochs = train_hparams.get("max_epochs")
    gpus = train_hparams.get("gpus")
    num_workers_for_dm = train_hparams.get("num_workers_for_dm")
    test_only = train_hparams.get("test_only")
    trainer_strategy = train_hparams.get("trainer_strategy")
    freeze_modalities = train_hparams.get("freeze_modalities")
    path_load_pretrained_imu_encoder = train_hparams.get("path_load_pretrained_imu_encoder")
    path_load_pretrained_audio_encoder = train_hparams.get("path_load_pretrained_audio_encoder")

    # Paths, etc.
    path_root_save_dir = f"{output_dir}/saved/{model_name}"
    if not os.path.exists(path_root_save_dir):
        os.makedirs(path_root_save_dir)
    target_modalities.sort()
    list_modalities = [source_modality] + target_modalities
    source_modality_initial = source_modality[0]
    target_modality_initials = "".join([m[0] for m in target_modalities])
    if source_modality == "imu":
        source_encoder_name = imu_encoder_name
    if source_modality == "audio":
        source_encoder_name = audio_encoder_name
    if source_modality == "text":
        source_encoder_name = text_encoder_name
    model_identifier = (
        f"{model_name}_s_{source_modality_initial}_t_{target_modality_initials}"
        + f"_se_{source_encoder_name}_w_{window_sec}"
    )
    if model_suffix != "":
        model_identifier += "_" + model_suffix
    else:
        model_identifier += "_%d" % (int(datetime.now().timestamp() % 10000))
    path_save_checkpoint = f"{path_root_save_dir}/{model_identifier}_best.ckpt"
    path_save_src_encoder = f"{path_root_save_dir}/{model_identifier}_src_encoder.pt"
    result_path = f"./results/{model_identifier}"
    configs["path_save_checkpoint"] = path_save_checkpoint

    # Initialize the data module
    dataset_params = {
        "window_sec": window_sec,
        "target_fps": target_fps,
        "list_modalities": list_modalities,
        "clean_narration_func": clean_narration_text,
        "filter_narration_func": filter_narration,
        "imu_sampling_rate": imu_sampling_rate,
    }

    if "text" in list_modalities:
        datamodule = Ego4dDataModule(
            batch_size=batch_size,
            num_workers=num_workers_for_dm,
            pin_memory=True,
            drop_last=True,
            dataset_params=dataset_params,
            split_num=configs["train_hparams"]["split_number"],
            data_path=configs["train_hparams"]["data_path"],
        )
    else:
        datamodule = UnsupEgo4dDataModule(
            batch_size=batch_size,
            num_workers=num_workers_for_dm,
            pin_memory=True,
            drop_last=True,
            dataset_params=dataset_params,
            split_num=configs["train_hparams"]["split_number"],
            data_path=configs["train_hparams"]["data_path"],
        )

    # Initialize encoder models
    text_encoder, video_encoder, imu_encoder = None, None, None
    modality_to_encoder = {}

    if "text" in list_modalities:
        # For now we only use a CLIP-based text model
        text_encoder = ClipPLModel(freeze=True)
        modality_to_encoder["text"] = text_encoder

    if "imu" in list_modalities:
        if imu_encoder_name == "pt":
            imu_encoder = PatchTransformer(
                size_embeddings=final_embedding_size,
                # num_heads=8,
                # num_layers=4,
                # num_patches=16,
                # patch_size=4,
                # hidden_size=512,
                # dropout=0.1,
            )
        elif imu_encoder_name == "ap":
            imu_encoder = AttentionPooledIMUEncoder(
                size_embeddings=final_embedding_size,
                # num_heads=8,
                # num_layers=4,
                # hidden_size=512,
                # dropout=0.1,
            )
        elif imu_encoder_name == "mw2":
            imu_encoder = MW2StackRNNPooling(size_embeddings=final_embedding_size) #
        elif imu_encoder_name == "senvt":
            imu_encoder = Ts.__dict__[configs["train_hparams"]["model_size"]](window_size=1000, in_chans=6, patch_size=configs["train_hparams"]["patch_size"])
            imu_encoder.head = torch.nn.Linear(imu_encoder.dim, 512)

        if path_load_pretrained_imu_encoder:
            # Load the parameters
            imu_encoder.load_state_dict(torch.load(path_load_pretrained_imu_encoder))
            logging.info("loaded pretrained imu model")

        modality_to_encoder["imu"] = imu_encoder

    if "video" in list_modalities:
        # For now we only use a CLIP-based image model as a video encoder
        video_encoder = ClipPLModel(freeze=True) if text_encoder is None else text_encoder
        video_encoder.video_encoder_name = video_encoder_name

        modality_to_encoder["video"] = video_encoder

    for modality in list_modalities:
        if modality in freeze_modalities:
            modality_to_encoder[modality].eval()
            logging.info(f"Freezing modality: {modality}", )
            modality_to_encoder[modality].freeze()

    # Initialize the training module for contrastive training
    model = MultimodalContrastiveLearningModule(
        modality_to_encoder=modality_to_encoder,
        source_modality=source_modality,
        target_modalities=target_modalities,
        output_dir=output_dir,
        learning_rate=configs["train_hparams"]["lr"],
    )
    
    logging.info(model)
    

    # Checkpoint settings
    checkpoint_callback_top3 = pl.callbacks.ModelCheckpoint(
        monitor="epoch_val_loss",
        dirpath=path_root_save_dir,
        filename=f"{model_identifier}" + "-{epoch:02d}-{epoch_val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    checkpoint_callback_latest = pl.callbacks.ModelCheckpoint(
        dirpath=path_root_save_dir,
        filename=f"{model_identifier}" + "-latest",
        save_top_k=1,  # Keep only the latest model
        monitor=None,  # No specific metric to monitor
        every_n_epochs=1,  # Save at the end of every epoch
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        # gpus=gpus,
        strategy=trainer_strategy,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        # limit_test_batches=limit_test_batches,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback_top3, checkpoint_callback_latest],
        logger=CSVLogger(save_dir="./"),
        #detect_anomaly=True
    )

    if not test_only:
        # Start training
        logging.info("Start training: [%s] ..." % path_save_checkpoint)
        if configs["train_hparams"]["path_load_from_checkpoint"]:
            model = MultimodalContrastiveLearningModule.load_from_checkpoint(
                configs["train_hparams"]["path_load_from_checkpoint"],
                modality_to_encoder=modality_to_encoder,
                source_modality=source_modality,
                target_modalities=target_modalities,
                output_dir=output_dir,
                learning_rate=configs["train_hparams"]["lr"],)
           
        trainer.fit(model, datamodule=datamodule)

        # Save the checkpoint & encoder to a temp folder
        torch.distributed.barrier()
        logging.info(f"Best checkpoint:{checkpoint_callback_top3.best_model_path}" )
        model = MultimodalContrastiveLearningModule.load_from_checkpoint(
            checkpoint_callback_top3.best_model_path,
            modality_to_encoder=modality_to_encoder,
            source_modality=source_modality,
            target_modalities=target_modalities,
            output_dir=output_dir,
            learning_rate=configs["train_hparams"]["lr"],
        )
        src_encoder = None
        if source_modality == "imu":
            src_encoder = model.imu_encoder
        elif source_modality == "audio":
            src_encoder = model.audio_encoder
        elif source_modality == "video":
            src_encoder = model.video_encoder
        torch.save(src_encoder.state_dict(), path_save_src_encoder)
    else:
        # Save the checkpoint & encoder to a temp folder
        # torch.distributed.barrier()
        # logging.info("Best checkpoint:", checkpoint_callback.best_model_path)
        # model = MultimodalContrastiveLearningModule.load_from_checkpoint(
        #     args.path_load_from_checkpoint,
        #     modality_to_encoder=modality_to_encoder,
        #     source_modality=source_modality,
        #     target_modalities=target_modalities,
        #     output_dir=output_dir,
        #     learning_rate=args.lr,
        # )
        if configs["train_hparams"]["use_egohos_best_pt"]:
            state_dict = torch.load('egohos_best.pt') 
            model.load_state_dict(state_dict, strict=False)
        else:
            model = MultimodalContrastiveLearningModule.load_from_checkpoint(
            configs["train_hparams"]["path_load_from_checkpoint"],
            modality_to_encoder=modality_to_encoder,
            source_modality=source_modality,
            target_modalities=target_modalities,
            output_dir=output_dir,
            learning_rate=configs["train_hparams"]["lr"],) 
            if configs["train_hparams"]["save_imu_encoder"]:
                torch.save(model.imu_encoder.state_dict(), output_dir+"/imu_encoder.pt")
        src_encoder = None
        if source_modality == "imu":
            src_encoder = model.imu_encoder
        elif source_modality == "audio":
            src_encoder = model.audio_encoder
        elif source_modality == "video":
            src_encoder = model.video_encoder
    # Test the performance
    logging.info("Start evaluating ...")
    # test_data = datamodule.get_dataset(
    #         "test",
    #         window_sample_rate=1.0,
    #         video_uid_sample_rate=configs["train_hparams"]["limit_test_batches"],  # 0.25,
    #         # max_n_windows_per_video=2,
    #     )
    
    metrics = evaluate(
        datamodule.get_dataset(
            "test",
            window_sample_rate=1.0,
            video_uid_sample_rate=configs["train_hparams"]["limit_test_batches"],  # 0.25,
            # max_n_windows_per_video=2,
        ),
        datamodule.collate_fn,
        model,
        source_modality,
        target_modalities,
        result_path,
        configs,
    )
    logging.info(metrics)
    print(metrics, file=codecs.open(f"{output_dir}/metrics.txt", "w", "utf-8"))
    return metrics


if __name__ == "__main__":

    parser = ArgumentParser()
    

    # Main parameters are defined in a YAML file
    parser.add_argument("--path_configs", default="./configs/train_contrastive/default.yaml")

    # Override-params for a quick resource allocation adjustment or for debugging purposes
    # If it is *not* None, the values in args override the values in the YAML file.
    # parser.add_argument("--gpus", default=None)
    # parser.add_argument("--num_workers_for_dm", default=None)
    # parser.add_argument("--max_epochs", default=None)
    # parser.add_argument("--path_load_pretrained_imu_encoder", default=None)
    # parser.add_argument("--path_load_from_checkpoint", default=None)
    # parser.add_argument("--split_number", default=1, type=int)
    # parser.add_argument("--lr", default=2e-4, type=float)
    args = parser.parse_args()

    # Load the YAML file
    with open(args.path_configs) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Override the configs with args, if requested
    # if args.gpus is not None:
    #     configs["train_hparams"]["gpus"] = int(args.gpus)
    # if args.num_workers_for_dm is not None:
    #     configs["train_hparams"]["num_workers_for_dm"] = int(args.num_workers_for_dm)
    # if args.max_epochs is not None:
    #     configs["train_hparams"]["max_epochs"] = int(args.max_epochs)
    # if args.test_only is not None:
    #     configs["train_hparams"]["test_only"] = eval(args.test_only)
    # if args.path_load_pretrained_imu_encoder is not None:
    #     configs["train_hparams"]["path_load_pretrained_imu_encoder"] = args.path_load_pretrained_imu_encoder
        
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    experiment_name = configs["model_hparams"]["model_name"]+"_s_"+configs["train_hparams"]["source_modality"][0]+"_t_"+"".join([m[0] for m in configs["train_hparams"]["target_modalities"]])+"_se_"+configs["model_hparams"]["imu_encoder_name"]+"_w_"+str(configs["model_hparams"]["window_sec"])+"_"+now.strftime("%Y-%m-%d_%H-%M-%S")
    if not configs["train_hparams"]["test_only"]:
        output_dir = os.path.join(f"experiments/train",experiment_name)
    else:
        output_dir = os.path.join(f"experiments/test",experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(output_dir)
    logging.info(args)
    logging.info(args)
    command = " ".join(sys.argv)
    logging.info(command)
    logging.info(configs)
    logging.info(
        f"PID:{os.getpid()}"
    )
    

    logging.info(configs)
    train(configs)
