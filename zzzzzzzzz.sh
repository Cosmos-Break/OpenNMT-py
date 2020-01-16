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


