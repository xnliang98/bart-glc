

# DATA_DIR=data/mc_final
task=csds_glc
# DATA_DIR=data/$task
DATA_DIR=data/csds_prompt

python src/run_summ_zh_demo.py \
    --model_name_or_path bart-base-chinese-cluecorpussmall \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/validation.json \
    --test_file $DATA_DIR/test.final.json \
    --do_train \
    --do_eval \
    --do_predict \
    --cache_dir cached_file/ \
    --preprocessing_num_workers 12 \
    --max_source_length 512 \
    --max_target_length 150 \
    --pad_to_max_length \
    --num_beams 3 \
    --ignore_pad_token_for_loss \
    --output_dir tmp/$task \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --label_smoothing_factor 0.0 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --num_train_epochs 5.0 \
    --max_steps -1 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "linear" \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --logging_steps 300 \
    --save_steps 300 \
    --eval_steps 300 \
    --overwrite_output_dir \
    --predict_with_generate