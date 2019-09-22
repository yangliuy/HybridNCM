
export CUDA_VISIBLE_DEVICES=1
repo_path="$PWD/../"
data_dir="$repo_path/demo_data/"
save_dir="$repo_path/model/"
mkdir -p $save_dir

python test.py \
    --test_src_file $data_dir/test.tok.lc.context \
    --test_trg_file $data_dir/test.tok.lc.response \
    --test_fact_file $data_dir/test.tok.lc.fact \
    --load_model_from $save_dir/model.bin \
    --save_decode_file $save_dir/decode-len30-beam5.txt \
    --decode_max_length 30 \
    --beam 5 \
    --vocab $save_dir/vocab.bin \
    --batch_size 32 \
    --cuda \
    --arch seq2seq
