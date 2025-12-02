for level in  'large' 'middle';do
echo $level
python auto-mask-fast.py \
    --video_path ./data/lerf_ovs/figurines/images \
    --output_dir output/figurines \
    --batch_size 40 \
    --detect_stride 10 \
    --level ${level}
python visulization.py \
    --video_path ./data/lerf_ovs/figurines/images \
    --output_dir output/figurines \
    --level ${level}
done