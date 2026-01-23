export MSWEA_COST_TRACKING="ignore_errors"

# Set vLLM server endpoint
export HOSTED_VLLM_API_BASE="http://localhost:8000/v1"

# Function to wait for vLLM server readiness
wait_for_vllm() {
    echo "Waiting for vLLM server at $HOSTED_VLLM_API_BASE to be ready..."
    while ! curl -s "$HOSTED_VLLM_API_BASE/models" > /dev/null; do
        sleep 10
    done
    echo "vLLM server is ready!"
}

# # Define model 1
# MODEL=openai/gpt-oss-120b
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $MODEL \
#   --tensor_parallel_size 8 \
#   --max-model-len 131072 \
#   --max-num-batched-tokens 10240 \
#   --max-num-seqs 128 \
#   --gpu-memory-utilization 0.85 \
#   --no-enable-prefix-caching &
# VLLM_PID=$!

# wait_for_vllm

# python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 1 --subset gym --split train

# kill $VLLM_PID
# wait $VLLM_PID

# Define model 2
MODEL=allenai/Olmo-3.1-32B-Instruct-DPO
# MODEL=Qwen/Qwen3-4B-Instruct-2507
# CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
#   --tensor_parallel_size 1 \
#   --max-model-len 131072 \
#   --max-num-batched-tokens 10240 \
#   --max-num-seqs 128 \
#   --gpu-memory-utilization 0.85 \
#   --no-enable-prefix-caching &
# VLLM_PID=$!

# wait_for_vllm

# sleep 15 * 60


python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 23 --subset lite --split dev --run-id "run-1"
python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 23 --subset lite --split dev --run-id "run-2"
python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 23 --subset lite --split dev --run-id "run-3"
python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 23 --subset lite --split dev --run-id "run-4"
python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 23 --subset lite --split dev --run-id "run-5"
python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 23 --subset lite --split dev --run-id "run-6"
python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 23 --subset lite --split dev --run-id "run-7"
python3 swebench.py --environment-class docker --model "hosted_vllm/$MODEL" --workers 23 --subset lite --split dev --run-id "run-8"

# kill $VLLM_PID
# wait $VLLM_PID
