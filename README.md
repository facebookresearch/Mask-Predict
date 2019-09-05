# Mask-Predict


### Download model 
Description | Dataset | Model
---|---|---
MASK-PREDICT | [WMT14 English-German] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_en_de.tar.gz)
MASK-PREDICT | [WMT14 German-English] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_de_en.tar.gz)
MASK-PREDICT | [WMT16 English-Romanian] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_en_ro.tar.gz)
MASK-PREDICT | [WMT16 Romanian-English] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_ro_en.tar.gz)
MASK-PREDICT | [WMT17 English-Chinese] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_en_zh.tar.gz)
MASK-PREDICT | [WMT17 Chinese-English] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_zh_en.tar.gz)

### Preprocess

text=PATH_YOUR_DATA

output_dir=PATH_YOUR_OUTPUT

src=source_language

tgt=target_language

model_path=PATH_TO_MASKPREDICT_MODEL_DIR

python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref $text/train --validpref $text/valid --testpref $text/test  --destdir ${output_dir}/data-bin  --workers 60  --srcdict ${model_path}/maskPredict_${src}_${tgt}/dict.${src}.txt --tgtdict ${model_path}/maskPredict_${src}_${tgt}/dict.${tgt}.txt

### Train


model_dir=PLACE_TO_SAVE_YOUR_MODEL

python train.py ${output_dir}/data-bin --arch bert_transformer_seq2seq --share-all-embeddings --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self --max-tokens 8192 --weight-decay 0.01 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512  --fp16 --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 0 --save-dir ${model_dir}

### Evaluation


python generate_cmlm.py ${output_dir}/data-bin  --path ${model_dir}/checkpoint_best_average.pt  --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 10  --decoding-strategy mask_predict

# License
MASK-PREDICT is CC-BY-NC 4.0.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

```bibtex
@inproceedings{ghazvininejad2019MaskPredict,
  title = {Mask-Predict: Parallel Decoding of Conditional Masked Language Models},
  author = {Marjan Ghazvininejad, Omer Levy, Yinhan Liu, Luke Zettlemoyer},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year = {2019},
}
```
