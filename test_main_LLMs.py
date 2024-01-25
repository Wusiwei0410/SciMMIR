import pytorch_lightning as pl
import torch
import clip
from PIL import Image

from src.lightning_classes import MMIR_LLMs_DataMoudle , MMIR_LLMs
from pytorch_lightning.callbacks import ModelCheckpoint
# from transformers import BertModel , BertTokenizer
import os
import argparse
from transformers import BertTokenizer, BertModel, BlipModel, BlipProcessor
from lavis.models import load_model_and_preprocess

def run(args):
    #fixed random seed
    pl.seed_everything(1234)


    checkpoint_callback = ModelCheckpoint(
        monitor="Validation/MRR_forward",
        filename="checkpoints--{Validation/MRR_forward:.4f}",
        save_top_k=args.save_top_k,
        mode="max",
    )

    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        callbacks = callbacks,
        accelerator="gpu",
        devices = args.gpus,
        # gpus = args.gpus,
        max_epochs = args.max_epochs,
        fast_dev_run=False,
        gradient_clip_val=args.gradient_clip_val,
        num_sanity_val_steps=0,
        check_val_every_n_epoch = 1,
    )

    DM = MMIR_LLMs_DataMoudle(
        config = args,
    )

    Model = MMIR_LLMs(
        config = args,
    )
    DM.setup( )

    # test
    DM.setup('test')
    trainer.test(model=Model, datamodule=DM ) #, ckpt_path = 'best'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for Paper MMIR model based on CLIP")
    parser.add_argument("--Use_BERT" , type = int , default = False)
    parser.add_argument("--image_type" , type = str , default = 'overall')

    parser.add_argument("--select_print" , type = str , default = 'print_all_setting')
    parser.add_argument("--score_method" , type = str , default = 'Matrix Dot Product')
    # parser.add_argument("--model_name" , type = str , default = 'llava_v1.5_7b')
    # parser.add_argument("--model_name" , type = str , default = 'mplug_owl2_llama2_7b')
    # parser.add_argument("--model_name" , type = str , default = 'kosmos')

    # parser.add_argument("--model_name" , type = str , default = 'llama-adapter')
    # parser.add_argument("--model_name" , type = str , default = 'blip2-flan-t5-xl')
    # parser.add_argument("--model_name" , type = str , default = 'blip2-flan-t5-xxl')
    # parser.add_argument("--model_name" , type = str , default = 'blip2-opt-2.7b')
    # parser.add_argument("--model_name" , type = str , default = 'blip2-opt-6.7b')
    parser.add_argument("--model_name" , type = str , default = 'fuyu-8b')
    
    parser.add_argument("--top_k" , type = int , default = 100)
    parser.add_argument("--embedding_size" , type = int , default = 512)
    parser.add_argument("--text_feature_len", type=int, default=768)
    parser.add_argument("--image_feature_len", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--save_top_k", type = int , default = 1)
    parser.add_argument("--max_epochs" , type = int ,default = 5)
    parser.add_argument("--gradient_clip_val", type = float , default= 5.0 )
    parser.add_argument("--weight_decay", type=float, default= 0.01)
    parser.add_argument("--alpha", type=float, default= 0.05)
    parser.add_argument("--lr_CLIP", type=float, default= 2e-5)
    parser.add_argument("--lr_BERT", type=float, default= 2e-5)
    parser.add_argument("--lr_Linner", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default= 1e-8)
    parser.add_argument("--gpus", type=list, default=[0])
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--fig_path", type=str, default="")
    parser.add_argument("--datasets_saved_path", type=str, default="")
    parser.add_argument("--num_warmup_steps", type=int, default=3155)
    parser.add_argument("--num_training_steps", type=int, default=31550)
    
    parser.add_argument("--train_batch_size" , type = int , default = 90)
    parser.add_argument("--figure_process_mat_num_valid", type=int, default=4)
    parser.add_argument("--figure_process_mat_num_test", type=int, default=4)

    parser.add_argument("--LLM_text_embedding_saved_path", type=str, default=f"./data/LLMs_Embeddings/{parser.parse_args().model_name}/")

    
    args = parser.parse_args()
    print(args)

    run(args)