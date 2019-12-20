MODEL_SNAPSHOT=NEW_IMGW_ADAM_bpe3000_dropout0.3_maxgrad0_step_1000.pt
MODEL_SNAPSHOT2=NEW_IMGW_ADAM_bpe3000_dropout0.3_maxgrad0_step_2000.pt
BPE_MERGE=3000
python translate.py -replace_unk -gpu 0 -model /home/mhxu/OpenNMT_fork/model_snapshots/${MODEL_SNAPSHOT} \
    /home/mhxu/OpenNMT_fork/model_snapshots/${MODEL_SNAPSHOT2} \
    -src '/home/mhxu/data/multi30k-dataset/data/task1/bpe'${BPE_MERGE}'/en-de/test_2016_flickr.lc.norm.tok.bpe.en' \
    -path_to_test_img_feats '/home/mhxu/data/features_resnet50/test_2016_flickr-resnet50-avgpool.npy' \
    --multimodal_model_type imgw \
    -output '/home/mhxu/OpenNMT_fork/model_snapshots/'${MODEL_SNAPSHOT}.test2016output
sed -r 's/(@@ )|(@@ ?$)//g' < '/home/mhxu/OpenNMT_fork/model_snapshots/'${MODEL_SNAPSHOT}.test2016output \
   > '/home/mhxu/OpenNMT_fork/model_snapshots/'${MODEL_SNAPSHOT}.test2016output.removebpe
perl tools/multi-bleu.perl '/home/mhxu/data/multi30k-dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.de' \
   < '/home/mhxu/OpenNMT_fork/model_snapshots/'${MODEL_SNAPSHOT}.test2016output.removebpe