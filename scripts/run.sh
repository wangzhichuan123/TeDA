# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
  echo "Usage: $0 <dataset> <backbone> <open_clip> <gpu_id>"
  exit 1
fi

DATASET=$1
BACKBONE=$2
OPEN_CLIP=$3
GPU_ID=$4

# 设置 CUDA_VISIBLE_DEVICES 环境变量
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Running $DATASET on GPU: $GPU_ID"

# 1. 提取 InternVL 特征
echo "Running internVL_feats_extract.py..."
python internVL_feats_extract.py --dataset "$DATASET" --backbone "$BACKBONE" --open_clip "$OPEN_CLIP"

# 2. 提取 CLIP 特征
echo "Running clip_feats_extract.py..."
python clip_feats_extract.py --dataset "$DATASET" --backbone "$BACKBONE" --open_clip "$OPEN_CLIP"

# 3. Run TeDA
echo "Running run_teda.py..."
python run_teda.py --dataset "$DATASET" --backbone "$BACKBONE" --open_clip "$OPEN_CLIP"

echo "All done!"