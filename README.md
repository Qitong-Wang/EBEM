# EBEM: An Enhanced Bi-encoder Model for Word Sense Disambiguation
This is the source code for EBEM.

## Dependencies
To run this code, you'll need the following libraries:
* nltk 3.7
* wandb 0.12.17
* matplotlib 3.5.2
* sacremoses 0.0.53
* transformers 4.19.2
* pytorch-lightning 1.6.4
* sentence-transformers 2.2.0
* pytorch-metric-learning 1.3.2

We used the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) for training and evaluating our model. Download the evaluation framework for convenient employment.

For WordNet Gloss Corpus (WNGC), we use [UFSAC](https://github.com/getalp/UFSAC).

For Sentence-BERT (SBERT), we use [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).

For Pre-trained Language Models (PLMs), we use [DeBERTa-base](https://huggingface.co/microsoft/deberta-base) and [DeBERTa-large](https://huggingface.co/microsoft/deberta-large).

## Train
For the base model, run:
```bash
python run_train.py \
       --framework EBEM \
       --model_name_or_path ./deberta-base \
       --sbert_name_or_path ./all-mpnet-base-v2 \
       --corpus_path ./WSD_Evaluation_Framework \
       --train_data Semcor \
       --val_data semeval2007 \
       --precision 32 \
       --max_epochs 5 \
       --num_workers 4 \
       --batch_size 256 \
       --max_context_seq_len 256 \
       --max_gloss_seq_len 48 \
       --circle_loss_m 0.5 \
       --circle_loss_gamma 32 \
       --learning_rate 1e-5 \
       --warmup_proportion 0.3 \
       --min_semantic_sim 0.1 \
       --polynomial_power 3.0 \
       --optimizer Adafactor \
       --enable_checkpointing 
```

For the large model, run:
```bash
python run_train.py \
       --framework EBEM \
       --model_name_or_path ./deberta-large \
       --sbert_name_or_path ./all-mpnet-base-v2 \
       --corpus_path ./WSD_Evaluation_Framework \
       --train_data Semcor \
       --val_data semeval2007 \
       --precision 32 \
       --max_epochs 5 \
       --num_workers 4 \
       --batch_size 256 \
       --max_context_seq_len 256 \
       --max_gloss_seq_len 48 \
       --circle_loss_m 0.5 \
       --circle_loss_gamma 32 \
       --learning_rate 3e-6 \
       --warmup_proportion 0.3 \
       --min_semantic_sim 0.2 \
       --polynomial_power 3.0 \
       --optimizer Adafactor \
       --enable_checkpointing 
```

For the large+ model, run:
```bash
python run_train.py \
       --framework EBEM \
       --model_name_or_path ./deberta-large \
       --sbert_name_or_path ./all-mpnet-base-v2 \
       --corpus_path ./WSD_Evaluation_Framework \
       --train_data Semcor-WNGT \
       --val_data semeval2007 \
       --precision 32 \
       --max_epochs 5 \
       --num_workers 4 \
       --batch_size 256 \
       --max_context_seq_len 256 \
       --max_gloss_seq_len 48 \
       --circle_loss_m 0.5 \
       --circle_loss_gamma 48 \
       --learning_rate 5e-6 \
       --warmup_proportion 0.3 \
       --min_semantic_sim 0.3 \
       --polynomial_power 3.0 \
       --optimizer Adafactor \
       --enable_checkpointing 
```

## Evaluate
To evaluate the model, run:
```bash
python run_eval.py \
       --model_name_or_path ./deberta-large \
       --sbert_name_or_path ./all-mpnet-base-v2 \
       --corpus_path ./WSD_Evaluation_Framework \
       --checkpoint_path ./checkpoints/CHECKPOINT_FILE_NAME.ckpt
```
