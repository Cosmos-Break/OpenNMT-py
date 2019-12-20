[2019-12-20 16:07:29,432 INFO] Translating shard 0.
*****************************************************************************
enterskjsdlkfjlaksdjf;alskdjf;laskdjf;laskdjf;laskdjf;alskdjf
<class 'onmt.decoders.ensemble.EnsembleEncoder'>
Traceback (most recent call last):
  File "translate.py", line 6, in <module>
    main()
  File "/home/mhxu/OpenNMT_fork/onmt/bin/translate.py", line 61, in main
    translate(opt)
  File "/home/mhxu/OpenNMT_fork/onmt/bin/translate.py", line 44, in translate
    multimodal_model_type=opt.multimodal_model_type
  File "/home/mhxu/OpenNMT_fork/onmt/translate/multimodaltranslator.py", line 106, in translate
    batch, data.src_vocabs, attn_debug, test_img_feats
  File "/home/mhxu/OpenNMT_fork/onmt/translate/multimodaltranslator.py", line 242, in translate_batch
    decode_strategy, img_feats)
  File "/home/mhxu/OpenNMT_fork/onmt/translate/multimodaltranslator.py", line 282, in _translate_batch_with_strategy
    src, enc_states, memory_bank, src_lengths = self._run_encoder(batch, img_feats)
  File "/home/mhxu/OpenNMT_fork/onmt/translate/multimodaltranslator.py", line 502, in _run_encoder
    src, img_feats, src_lengths)
  File "/home/mhxu/anaconda3/envs/OpenNMT/lib/python3.6/site-packages/torch/nn/modules/module.py", line 547, in __call__
    result = self.forward(*input, **kwargs)
TypeError: forward() takes from 2 to 3 positional arguments but 4 were given
Use of uninitialized value $length_reference in numeric eq (==) at tools/multi-bleu.perl line 148


PRED AVG SCORE: -0.2502, PRED PPL: 1.2843
BLEU = 38.63, 70.3/46.0/32.4/23.0 (BP=0.980, ratio=0.980, hyp_len=11860, ref_len=12103)
[2019-12-19 12:16:16,213 INFO] Translating shard 0.
/content/drive/Shared drives/Aria/OpenNMT-Fork/onmt/translate/multimodaltranslator.py:497: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  src, Variable(img_feats, volatile=True), src_lengths)
/content/drive/Shared drives/Aria/OpenNMT-Fork/onmt/translate/multimodaltranslator.py:301: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  img_feats = Variable(img_feats.repeat(self.beam_size, 1), volatile=True)




#best：
!python train_mmod.py -data data/m30kbpe3000 -save_model model_snapshots/IMGW_ADAM_bpe3000_dropout0.3 -gpuid 0 -layers 6 -position_encoding -max_generator_batches 32 -batch_size 40   -accum_count 2 -epochs 250 -rnn_size 512 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type transformer -dropout 0.3 -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy' -optim adam  -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -decoder_type transformer  --multimodal_model_type imgw 
# IMGD_ADAM_acc_69.33_ppl_7.84_e25.pt
!python train_mmod.py -data data/m30kbpe3000 -save_model model_snapshots/IMGW_ADAM_bpe3000_dropout0.3_maxgrad5 -gpuid 0 -layers 6 -position_encoding -max_generator_batches 32 -batch_size 40   -accum_count 2 -epochs 250 -rnn_size 512 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type transformer -dropout 0.3 -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy' -optim adam  -decay_method noam -warmup_steps 8000 -learning_rate 2  -param_init_glorot -label_smoothing 0.1 -decoder_type transformer  --multimodal_model_type imgw 


sed -r 's/(@@ )|(@@ ?$)//g'


!python train_mmod.py -data data/m30kbpe3000 -save_model model_snapshots/IMGW_ADAM_bpe3000_dropout0.3 -gpuid 0 -layers 6 -position_encoding -max_generator_batches 32 -batch_size 40   -batch_type tokens -normalization tokens -accum_count 2 -train_steps 200000 -valid_steps 10000 -save_checkpoint_steps 10000 -rnn_size 512 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type transformer -dropout 0.3 -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy' -optim adam  -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -decoder_type transformer  --multimodal_model_type imgw 


python  train.py -data data/m30kbpe3000 -save_model model_snapshots/NEW_IMGW_ADAM_bpe3000_dropout0.3 \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.3 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 5 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 1000 \
        -world_size 1 -gpu_ranks 0 \
        -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' \
        -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy' \
        --multimodal_model_type imgw 

python  train.py -data data/m30kbpe3000 -save_model model_snapshots/NEW_IMGW_ADAM_bpe3000_dropout0.3 \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.3 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 5 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 1000 \
        -world_size 1 -gpu_ranks 0 \
        -path_to_train_img_feats '/home/mhxu/data/features_resnet50/train-resnet50-avgpool.npy' \
        -path_to_valid_img_feats '/home/mhxu/data/features_resnet50/val-resnet50-avgpool.npy' \
        --multimodal_model_type imgw 


%%bash
MODEL_SNAPSHOT=
sed -r 's/(@@ )|(@@ ?$)//g' < example/${MODEL_SNAPSHOT} > example/${MODEL_SNAPSHOT}.removebpe

./multeval.sh eval --refs example/test_2016_flickr.lc.norm.tok.de \
                   --hyps-baseline example/IMGW_ADAM6_acc_77.44_ppl_3.06_e44.pt.test2016output \
                   --meteor.language de

./multeval.sh eval --refs example/test_2016_flickr.lc.norm.tok.de  --hyps-baseline example/IMGW_ADAM*.removebpe  --meteor.language de
example/IMGW_ADAM1000_acc_73.14_ppl_5.57_e23.pt.test2016output

!python train_mmod.py -epochs 250 -data data/m30kbpe10000 -save_model model_snapshots/IMGW_ADAM10000 -gpuid 0 -layers 6 -rnn_size 512 -word_vec_size 512 -encoder_type transformer -decoder_type transformer -position_encoding -max_generator_batches 2 -dropout 0.5 -batch_size 40 -accum_count 2 -optim adam -decay_method noam -warmup_steps 8000 -learning_rate 2  -max_grad_norm 0 -param_init 0  -param_init_glorot  -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy'  --multimodal_model_type imgw 

!python train_mmod.py -epochs 250 -data data/m30kbpe10000 -save_model model_snapshots/IMGW_ADAM10000 -gpuid 0 -layers 6 -rnn_size 512 -word_vec_size 512 -encoder_type transformer -decoder_type transformer -position_encoding -max_generator_batches 2 -dropout 0.1 -batch_size 40 -accum_count 2 -optim adam  -decay_method noam -warmup_steps 8000 -learning_rate 2  -max_grad_norm 0 -param_init 0  -param_init_glorot  -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy'  --multimodal_model_type imgw 


%%bash
MODEL_SNAPSHOT=IMGW_ADAM6_acc_77.44_ppl_3.06_e44.pt
/content/drive/'Shared drives'/Aria/multeval-0.5.1/multeval.sh eval --refs '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe500/en-de/test_2016_flickr.lc.norm.tok.bpe.de' --hyps-baseline '/content/drive/Shared drives/Aria/OpenNMTMeMAD/model_snapshots/'${MODEL_SNAPSHOT}.test2016output --meteor.language de

%%bash
MODEL_SNAPSHOT=IMGW_ADAM6_acc_77.44_ppl_3.06_e44.pt
python translate_mmod_finetune.py -gpu 0 -model '/content/drive/Shared drives/Aria/OpenNMTMeMAD/model_snapshots/'${MODEL_SNAPSHOT} -src '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe500/en-de/test_2016_flickr.lc.norm.tok.bpe.en'  -path_to_test_img_feats '/content/drive/Shared drives/Aria/features_resnet50/test_2016_flickr-resnet50-avgpool.npy' --multimodal_model_type imgw -mmod_use_hidden -output '/content/drive/Shared drives/Aria/OpenNMTMeMAD/model_snapshots/'${MODEL_SNAPSHOT}.test2016output
perl tools/multi-bleu.perl '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe500/en-de/test_2016_flickr.lc.norm.tok.bpe.de' < '/content/drive/Shared drives/Aria/OpenNMTMeMAD/model_snapshots/'${MODEL_SNAPSHOT}.test2016output
# creating a Sequential generator
# Loading model parameters.
# average src size 23.075 1000
# PRED AVG SCORE: -0.2264, PRED PPL: 1.2540
# BLEU = 42.71, 66.7/49.8/40.2/32.4 (BP=0.937, ratio=0.939, hyp_len=22331, ref_len=23789)
-----------------------------------------------------------------------------------------------------------------------

# Preprocess:
!python preprocess.py -train_src '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe3000/en-de/train.lc.norm.tok.bpe.en' -train_tgt '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe3000/en-de/train.lc.norm.tok.bpe.de' -valid_src '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe3000/en-de/val.lc.norm.tok.bpe.en' -valid_tgt '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe3000/en-de/val.lc.norm.tok.bpe.de' -src_vocab '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe3000/en-de/vocab.en' -tgt_vocab '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/bpe3000/en-de/vocab.de'  -save_data data/m30kbpe3000

!python preprocess.py -train_src '/home/mhxu/data/multi30k-dataset/data/task1/bpe3000/en-de/train.lc.norm.tok.bpe.en' -train_tgt '/home/mhxu/data/multi30k-dataset/data/task1/bpe3000/en-de/train.lc.norm.tok.bpe.de' -valid_src '/home/mhxu/data/multi30k-dataset/data/task1/bpe3000/en-de/val.lc.norm.tok.bpe.en' -valid_tgt '/home/mhxu/data/multi30k-dataset/data/task1/bpe3000/en-de/val.lc.norm.tok.bpe.de' -src_vocab '/home/mhxu/data/multi30k-dataset/data/task1/bpe3000/en-de/vocab.en' -tgt_vocab '/home/mhxu/data/multi30k-dataset/data/task1/bpe3000/en-de/vocab.de'  -save_data data/m30kbpe3000


# Translate:
!python translate_mmod_finetune.py -report_bleu -report_rouge -gpu 0 -model '/content/drive/Shared drives/Aria/OpenNMTMeMAD/model_snapshots/IMGW_ADAM_acc_67.39_ppl_13.73_e153.pt' -src '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.en'  -path_to_test_img_feats '/content/drive/Shared drives/Aria/features_resnet50/test_2016_flickr-resnet50-avgpool.npy' --multimodal_model_type imgw -mmod_use_hidden -output '/content/drive/Shared drives/Aria/OpenNMTMeMAD/model_snapshots/IMGW_ADAM_acc_67.39_ppl_13.73_e153.pt.test2016output'

!python translate_mmod_finetune.py -src ../multi30k-dataset/toke10000/en_de/test_2016_flickr.norm.tok.lc.bpe10000.en -gpu 0 -model ../Model_snapshots/en_de_16gg19_model_snapshots/model_snapshots_16en-de_no_cover_tf4/${MODEL_SNAPSHOT} -path_to_test_img_feats ../16_vgg19_bn_cnn_features/test_2016_flickr-resnet50-avgpool.npy -output ../Model_snapshots/en_de_16gg19_model_snapshots/model_snapshots_16en-de_no_cover_tf4/${MODEL_SNAPSHOT}.translations-test2016 --multimodal_model_type imgw

# Evaluate:
!perl tools/multi-bleu.perl '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.de' < '/content/drive/Shared drives/Aria/OpenNMTMeMAD/model_snapshots/IMGW_ADAM_acc_67.39_ppl_13.73_e153.pt.test2016output'

!MODEL_SNAPSHOT=IMGD_ADAM_acc_69.33_ppl_7.84_e40.pt
!python translate_mm.py -src '/content/drive/Shared drives/Aria/multi30k-dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.en' -model model_snapshots/IMGD_ADAM_acc_69.33_ppl_7.84_e40.pt -path_to_test_img_feats '/content/drive/Shared drives/Aria/features_resnet50/test_2016_flickr-resnet50-avgpool.npy' -output model_snapshots/IMGD_ADAM_acc_69.33_ppl_7.84_e40.pt.translations-test2016



python3 train_mmod.py -data ../data/m30k_en_de/m30k -save_model ../Model_snapshots/en_de_16gg19_model_snapshots/model_snapshots_16en-de_no_cover_tf4/MNMT-imgw_ADAM -gpuid 1 -layers 6 -position_encoding -max_generator_batches 2 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2  -epochs 25 -rnn_size 512 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type transformer -dropout 0.1 -path_to_train_img_feats ../16_vgg19_bn_cnn_features/train-resnet50-avgpool.npy -path_to_valid_img_feats ../16_vgg19_bn_cnn_featuresal-resnet50-avgpool.npy -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -decoder_type transformer --multimodal_model_type imgw
1:
!python train_mmod.py -data data/m30k -save_model model_snapshots/IMGW_ADAM -gpuid 0 -layers 6 -position_encoding -max_generator_batches 2 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -epochs 25 -rnn_size 512 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type transformer -dropout 0.1 -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy' -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -decoder_type transformer  --multimodal_model_type imgw 

2:
!python train_mmod.py -data data/m30k -save_model model_snapshots/IMGW_ADAM -gpuid 0 -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding -max_generator_batches 2 -dropout 0.1 -batch_size 40  -batch_size 4096 -batch_type tokens -normalization tokens -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2  -max_grad_norm 0 -param_init 0  -param_init_glorot -epochs 25 -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy'  --multimodal_model_type imgw 




3:
!python train_mmod.py -data data/m30k -save_model model_snapshots/IMGW_ADAM -gpuid 0 -layers 6 -position_encoding -max_generator_batches 32 -batch_size 40   -accum_count 4 -epochs 250 -rnn_size 512 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type transformer -dropout 0.2 -path_to_train_img_feats '/content/drive/Shared drives/Aria/features_resnet50/train-resnet50-avgpool.npy' -path_to_valid_img_feats '/content/drive/Shared drives/Aria/features_resnet50/val-resnet50-avgpool.npy' -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -decoder_type transformer  --multimodal_model_type imgw 





Loading train dataset from data/m30k.train.1.pt, number of examples: 29000
data_type:  text
 * vocabulary size. source = 10212; target = 18726
Building model...
creating a Sequential generator
MultiModalNMTModel(
  (encoder): MultiModalTransformerEncoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(10212, 512, padding_idx=1)
        )
        (pe): PositionalEncoding(
          (dropout): Dropout(p=0.1)
        )
      )
    )
    (transformer): ModuleList(
      (0): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm): LayerNorm(
        )
        (dropout): Dropout(p=0.1)
      )
      (1): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm): LayerNorm(
        )
        (dropout): Dropout(p=0.1)
      )
      (2): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm): LayerNorm(
        )
        (dropout): Dropout(p=0.1)
      )
      (3): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm): LayerNorm(
        )
        (dropout): Dropout(p=0.1)
      )
      (4): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm): LayerNorm(
        )
        (dropout): Dropout(p=0.1)
      )
      (5): TransformerEncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm): LayerNorm(
        )
        (dropout): Dropout(p=0.1)
      )
    )
    (layer_norm): LayerNorm(
    )
    (img_to_emb): Linear(in_features=2048, out_features=512, bias=True)
  )
  (decoder): TransformerDecoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(18726, 512, padding_idx=1)
        )
        (pe): PositionalEncoding(
          (dropout): Dropout(p=0.1)
        )
      )
    )
    (transformer_layers): ModuleList(
      (0): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm_1): LayerNorm(
        )
        (layer_norm_2): LayerNorm(
        )
        (drop): Dropout(p=0.1)
      )
      (1): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm_1): LayerNorm(
        )
        (layer_norm_2): LayerNorm(
        )
        (drop): Dropout(p=0.1)
      )
      (2): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm_1): LayerNorm(
        )
        (layer_norm_2): LayerNorm(
        )
        (drop): Dropout(p=0.1)
      )
      (3): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm_1): LayerNorm(
        )
        (layer_norm_2): LayerNorm(
        )
        (drop): Dropout(p=0.1)
      )
      (4): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm_1): LayerNorm(
        )
        (layer_norm_2): LayerNorm(
        )
        (drop): Dropout(p=0.1)
      )
      (5): TransformerDecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (context_attn): MultiHeadedAttention(
          (linear_keys): Linear(in_features=512, out_features=512, bias=True)
          (linear_values): Linear(in_features=512, out_features=512, bias=True)
          (linear_query): Linear(in_features=512, out_features=512, bias=True)
          (sm): Softmax()
          (dropout): Dropout(p=0.1)
          (final_linear): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (layer_norm): LayerNorm(
          )
          (dropout_1): Dropout(p=0.1, inplace)
          (relu): ReLU(inplace)
          (dropout_2): Dropout(p=0.1)
        )
        (layer_norm_1): LayerNorm(
        )
        (layer_norm_2): LayerNorm(
        )
        (drop): Dropout(p=0.1)
      )
    )
    (layer_norm): LayerNorm(
    )
  )
  (generator): Sequential(
    (0): Linear(in_features=512, out_features=18726, bias=True)
    (1): LogSoftmax()
  )
)
* number of parameters: 69612326
encoder:  25192960
decoder:  44419366
Making optimizer for training.
Stage 1: Keys after executing optim.set_parameters(model.parameters())
optim.optimizer.state_dict()['state'] keys: 
optim.optimizer.state_dict()['param_groups'] elements: 
optim.optimizer.state_dict()['param_groups'] element: {'lr': 2.0, 'betas': [0.9, 0.998], 'eps': 1e-09, 'weight_decay': 0, 'params': [139818575092456, 139818575093176, 139818575093256, 139818575093416, 139818575093496, 139818575093656, 139818574778440, 139818574778760, 139818574778840, 139818574779080, 139818574779160, 139818574779320, 139818574779400, 139818574779560, 139818574779640, 139818574779960, 139818574780040, 139818574780360, 139818574780440, 139818574780600, 139818574780680, 139818574780840, 139818574780920, 139818574781240, 139818574781320, 139818574781560, 139818574781640, 139818574781800, 139818574781880, 139818574782040, 139818574782120, 139818574295112, 139818574295192, 139818574295512, 139818574295592, 139818574295752, 139818574295832, 139818574295992, 139818574296072, 139818574296392, 139818574296472, 139818574296712, 139818574296792, 139818574296952, 139818574297032, 139818574297192, 139818574297272, 139818574297592, 139818574297672, 139818574297992, 139818574298072, 139818574298232, 139818574298312, 139818574298472, 139818574298552, 139818574298872, 139818574298952, 139818574332056, 139818574332136, 139818574332296, 139818574332376, 139818574332536, 139818574332616, 139818574332936, 139818574333016, 139818574333336, 139818574333416, 139818574333576, 139818574333656, 139818574333816, 139818574333896, 139818574334216, 139818574334296, 139818574334536, 139818574334616, 139818574334776, 139818574334856, 139818574335016, 139818574335096, 139818574335416, 139818574335496, 139818574335816, 139818574335896, 139818574360728, 139818574360808, 139818574360968, 139818574361048, 139818574361368, 139818574361448, 139818574361688, 139818574361768, 139818574361928, 139818574362008, 139818574362168, 139818574362248, 139818574362568, 139818574362648, 139818574362968, 139818574363048, 139818574363208, 139818574363288, 139818574363448, 139818574364088, 139818574364168, 139818574364328, 139818574364408, 139818574364568, 139818574397512, 139818574397832, 139818574397912, 139818574398072, 139818574398152, 139818574398312, 139818574398392, 139818574398552, 139818574398632, 139818574398952, 139818574399032, 139818574399272, 139818574399352, 139818574399512, 139818574399592, 139818574399752, 139818574399832, 139818574400152, 139818574400232, 139818574400392, 139818574400472, 139818574400712, 139818574400872, 139818574401112, 139818574401192, 139818574401352, 139818574401432, 139818574434616, 139818574434696, 139818574434856, 139818574434936, 139818574435096, 139818574435176, 139818574435336, 139818574435416, 139818574435736, 139818574435816, 139818574436056, 139818574436136, 139818574436296, 139818574436376, 139818574436536, 139818574436616, 139818574436936, 139818574437016, 139818574437176, 139818574437256, 139818574437496, 139818574437656, 139818574437896, 139818574437976, 139818574438136, 139818574438216, 139818574467304, 139818574467384, 139818574467544, 139818574467624, 139818574467784, 139818574467864, 139818574468024, 139818574468104, 139818574468424, 139818574468504, 139818574468744, 139818574468824, 139818574468984, 139818574469064, 139818574469224, 139818574469304, 139818574469624, 139818574469704, 139818574469864, 139818574469944, 139818574470184, 139818574470344, 139818574470584, 139818574470664, 139818574470824, 139818574470904, 139818574499992, 139818574500072, 139818574500232, 139818574500312, 139818574500472, 139818574500552, 139818574500712, 139818574500792, 139818574501112, 139818574501192, 139818574501432, 139818574501512, 139818574501672, 139818574501752, 139818574501912, 139818574501992, 139818574502312, 139818574502392, 139818574502552, 139818574502632, 139818574502872, 139818574503032, 139818574503272, 139818574503352, 139818574503512, 139818574503592, 139818574532680, 139818574532760, 139818574532920, 139818574533000, 139818574533160, 139818574533240, 139818574533400, 139818574533480, 139818574533800, 139818574533880, 139818574534120, 139818574534200, 139818574534360, 139818574534440, 139818574534600, 139818574534680, 139818574535000, 139818574535080, 139818574535240, 139818574535320, 139818574535560, 139818574535720, 139818574535960, 139818574536040, 139818574536200, 139818574536280, 139818574536600, 139818574049352, 139818574049512, 139818574049592, 139818574049752, 139818574049832, 139818574049992, 139818574050072, 139818574050392, 139818574050472, 139818574050712, 139818574050792, 139818574050952, 139818574051032, 139818574051192, 139818574051272, 139818574051592, 139818574051672, 139818574051832, 139818574051912, 139818574052392, 139818574052152, 139818574052472, 139818574052552]}
Loading valid dataset from data/m30k.valid.1.pt, number of examples: 1014
valid 1013 (1014, 2048)

Start training...
 * number of epochs: 25, starting from Epoch 1
 * batch size: 40