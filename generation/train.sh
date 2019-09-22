
export CUDA_VISIBLE_DEVICES=1
repo_path="$PWD/../"
data_dir="$repo_path/demo_data/"
save_dir="$repo_path/model/"
mkdir -p $save_dir

python3 train.py \
    --train_src_file $data_dir/train_demo.tok.clean.lc.context \
    --train_trg_file $data_dir/train_demo.tok.clean.lc.response \
    --train_fact_file $data_dir/train_demo.tok.clean.lc.fact \
    --dev_src_file $data_dir/dev.tok.lc.context \
    --dev_trg_file $data_dir/dev.tok.lc.response \
    --dev_fact_file $data_dir/dev.tok.lc.fact \
    --test_src_file $data_dir/test.tok.lc.context \
    --test_trg_file $data_dir/test.tok.lc.response \
    --test_fact_file $data_dir/test.tok.lc.fact \
    --save_model_to $save_dir/model \
    --vocab $save_dir/vocab.bin \
    --batch_size 32 \
    --embed_size 128 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --hidden_size 128 \
    --lr 0.001 \
    --bidirectional \
    --valid_interval 500 \
    --patience 5 \
    --save_model_after 0 \
    --cuda \
    --arch seq2seq
