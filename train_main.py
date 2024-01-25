import pytorch_lightning as pl
import torch
import clip
from PIL import Image

from src.lightning_classes import SciMMIR_FT_DataMoudle , SciMMIR_FT
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse
from transformers import BertTokenizer, BertModel, BlipModel, BlipProcessor
from lavis.models import load_model_and_preprocess
from math import floor

def run(args):
    #fixed random seed
    pl.seed_everything(1234)
    if args.Use_BERT == True:
        logs_path = os.path.join("./checkpoints/" , f'MMIR_BERT_{args.model_name}_{args.image_type}')
    else:
        logs_path = os.path.join("./checkpoints/" , f'MMIR_{args.model_name}_{args.image_type}')
    if os.path.exists(logs_path) == False:
        os.makedirs(logs_path)
    
    if os.path.exists(args.result_save_path) == False:
        os.makedirs(args.result_save_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model_name == 'CLIP':
        model, preprocess = clip.load("ViT-B/32", device=device)
    elif args.model_name == 'BLIP':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", size = args.image_size)
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").cuda().float()
        preprocess = processor
    elif args.model_name == 'BLIP-large':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", size = args.image_size)
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large").cuda().float()
        preprocess = processor
    if args.Use_BERT == True:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERT_model = BertModel.from_pretrained("bert-base-uncased")
    else:
        tokenizer = clip.tokenize
    # lightningmodule checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="Validation/MRR_forward",
        dirpath=logs_path,
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

    DM = SciMMIR_FT_DataMoudle(
        preprocess = preprocess,
        tokenizer = tokenizer,
        config = args,
    )

    if args.Use_BERT == True:
        Model = SciMMIR_FT(
            config = args,
            model = model,
            BERT_model = BERT_model,
        )
    else:
        Model = SciMMIR_FT(
            config = args,
            model = model,
        )
    DM.setup( )
    

    # Training Model
    trainer.fit(model=Model, datamodule=DM)

    # test
    DM.setup('test')
    trainer.test(model=Model, datamodule=DM , ckpt_path = 'best')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for SciMMIR model")
    parser.add_argument("--Use_BERT" , type = int , default = False)
    parser.add_argument("--image_type" , type = str , default = 'overall')
    # parser.add_argument("--image_type" , type = str , default = 'fig_architecture')
    # parser.add_argument("--image_type" , type = str , default = 'fig_illustration')
    # parser.add_argument("--image_type" , type = str , default = 'fig_result')
    # parser.add_argument("--image_type" , type = str , default = 'table_result')
    # parser.add_argument("--image_type" , type = str , default = 'table_parameter')

    parser.add_argument("--select_print" , type = str , default = 'print_all_setting')
    parser.add_argument("--score_method" , type = str , default = 'Matrix Dot Product')
    parser.add_argument("--model_name" , type = str , default = 'CLIP')
    parser.add_argument("--training_data_len" , type = int , default = 504731)
    parser.add_argument("--top_k" , type = int , default = 100)
    parser.add_argument("--embedding_size" , type = int , default = 512)
    parser.add_argument("--text_feature_len", type=int, default=768)
    parser.add_argument("--image_feature_len", type=int, default=512)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--train_batch_size" , type = int , default = 210)
    parser.add_argument("--save_top_k", type = int , default = 1)
    parser.add_argument("--max_epochs" , type = int ,default = 5)
    parser.add_argument("--text_process_num" , type = int , default = 1000)
    parser.add_argument("--figure_process_num" , type = int , default = 100)
    parser.add_argument("--gradient_clip_val", type = float , default= 5.0 )
    parser.add_argument("--weight_decay", type=float, default= 0.01)
    parser.add_argument("--alpha", type=float, default= 0.05)
    parser.add_argument("--lr_CLIP", type=float, default= 2e-5)
    parser.add_argument("--lr_BERT", type=float, default= 2e-5)
    parser.add_argument("--lr_Linner", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default= 1e-8)
    parser.add_argument("--gpus", type=list, default=[0])
    parser.add_argument("--context_length", type=int, default=77)
    parser.add_argument("--image_size", type=int, default=224) #384 
    parser.add_argument("--datasets_saved_path", type=str, default="./data/")
    parser.add_argument("--result_save_path", type=str, default="./data/result/")
    parser.add_argument("--num_warmup_steps", type=int, default= floor(floor(parser.parse_args().training_data_len / parser.parse_args().train_batch_size) * parser.parse_args().max_epochs / 10))
    parser.add_argument("--num_training_steps", type=int, default=floor(parser.parse_args().training_data_len / parser.parse_args().train_batch_size) * parser.parse_args().max_epochs)#12020
    parser.add_argument("--figure_process_mat_num_valid", type=int, default=4)
    parser.add_argument("--figure_process_mat_num_test", type=int, default=4)

    parser.add_argument("--text_2_image_index_valid", type=str, default=f"{parser.parse_args().model_name}_torch_data_valid/text_2_image_index.json")
    parser.add_argument("--text_2_image_index_test", type=str, default=f"{parser.parse_args().model_name}_torch_data_test/text_2_image_index.json")
    parser.add_argument("--figure_process_mat_path_valid", type=str, default=f"{parser.parse_args().model_name}_torch_data_valid/")
    parser.add_argument("--figure_process_mat_path_test", type=str, default=f"{parser.parse_args().model_name}_torch_data_test/")
    
    if parser.parse_args().Use_BERT == True:
        model_name = "BERT"
    else:
        model_name = parser.parse_args().model_name
    parser.add_argument("--text_2_index_valid", type=str, default=f"{model_name}_torch_data_valid/text_index.json")
    parser.add_argument("--text_2_index_test", type=str, default=f"{model_name}_torch_data_test/text_index.json")
    parser.add_argument("--text_process_mat_path_valid", type=str, default=f"{model_name}_torch_data_valid/")
    parser.add_argument("--text_process_mat_path_test", type=str, default=f"{model_name}_torch_data_test/")
    
    args = parser.parse_args()
    print(args)

    run(args)
