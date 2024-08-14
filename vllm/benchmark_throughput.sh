source /opt/intel/oneapi/2024.0/oneapi-vars.sh --force
source /opt/intel/1ccl-wks/setvars.sh --force  # use oneCCL


export MODEL="/opt/Meta-Llama-3-8B-Instruct"

export CCL_WORKER_COUNT=2 ## 2 maybe means 2*A770
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
#export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}
export ZE_AFFINITY_MASK=0,1  ## 0,1,2,3 means 4*A770

for n in $(seq 8 2 20); do
    echo "Model= $MODEL RATE= 0.7 N= $n..."
    python3 ./benchmark_throughput.py \
        --backend vllm \
        --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
        --model $MODEL \
        --num-prompts 100 \
        --seed 42 \
        --trust-remote-code \
        --enforce-eager \
        --dtype float16 \
        --device xpu \
        --load-in-low-bit sym_int4 \
        --gpu-memory-utilization 0.7 \
        --max-num-seqs $n \
        --tensor-parallel-size 2  ## 2 means 2*A770
done
sleep 10
exit 0
