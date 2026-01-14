export MSWEA_COST_TRACKING="ignore_errors"

# Set vLLM server endpoint
export HOSTED_VLLM_API_BASE="http://localhost:8000/v1"

# Set Enroot Cache Folder
export ENROOT_CACHE_PATH="/mnt/weka/shrd/k2pta/swe-agent-enroot-images/"

# Define model
# MODEL=mistralai/Devstral-Small-2-24B-Instruct-2512
MODEL=openai/gpt-oss-120b

python3 swebench.py --environment-class enroot --model "hosted_vllm/$MODEL" --workers 1