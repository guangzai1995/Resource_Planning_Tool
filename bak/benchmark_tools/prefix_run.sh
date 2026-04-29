#!/bin/bash

# 初始化默认值
BACKEND=""
HOST=""
PORT=""
TOKENIZER=""
EPOCHS=""
PARALLEL_NUM=""
PROMPT_TOKENS=""
OUTPUT_TOKENS=""
SERVED_MODEL_NAME=""
API_KEY=""
BENCHMARK_CSV=""
ENABLE_PREFIX_CACHING="False"
PREFIX_CACHING_NUM="0"

# 解析命名参数（核心修改）
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;        
        --host) HOST="$2"; shift 2 ;;                
        --port) PORT="$2"; shift 2 ;;                
        --tokenizer) TOKENIZER="$2"; shift 2 ;;        
        --epochs) EPOCHS="$2"; shift 2 ;;            
        --parallel-num) PARALLEL_NUM="$2"; shift 2 ;;    
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;    
        --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;    
        --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;    
        --api-key) API_KEY="$2"; shift 2 ;;            
        --benchmark-csv) BENCHMARK_CSV="$2"; shift 2 ;;    
        --enable-prefix-caching) ENABLE_PREFIX_CACHING="$2"; shift 2 ;;    
        --prefix-caching-num) PREFIX_CACHING_NUM="$2"; shift 2 ;;    
        *) echo "错误：未知参数 $1"; exit 1 ;;        
    esac
 done

# 执行基准测试（保留条件参数逻辑）
python benchmark_parallel.py \
    --backend "$BACKEND" \
    --host "$HOST" \
    --port "$PORT" \
    --tokenizer "$TOKENIZER" \
    --epochs "$EPOCHS" \
    --parallel-num $PARALLEL_NUM \
    --prompt-tokens $PROMPT_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --served-model-name "$SERVED_MODEL_NAME" \
    --api-key "$API_KEY" \
    --num-scheduler-steps 1 \
    --benchmark-csv "$BENCHMARK_CSV" \
    $(if [ "$ENABLE_PREFIX_CACHING" = "True" ]; then echo "--enable-prefix-caching $ENABLE_PREFIX_CACHING --prefix-caching-num $PREFIX_CACHING_NUM"; fi)
# #!/bin/bash

# # 从命令行参数获取所有参数值
# BACKEND=$1
# HOST=$2
# PORT=$3
# TOKENIZER=$4
# EPOCHS=$5
# PARALLEL_NUM=$6
# PROMPT_TOKENS=$7
# OUTPUT_TOKENS=$8
# SERVED_MODEL_NAME=$9
# API_KEY=${10}
# BENCHMARK_CSV=${11}
# ENABLE_PREFIX_CACHING=${12}
# PREFIX_CACHING_NUM=${13}


# python benchmark_parallel.py --backend "$BACKEND" \
#       	--host "$HOST" \
# 	--port "$PORT" \
# 	--tokenizer "$TOKENIZER"  \
# 	--epochs "$EPOCHS" \
# 	--parallel-num "$PARALLEL_NUM" \
# 	--prompt-tokens "$PROMPT_TOKENS"  \
# 	--output-tokens "$OUTPUT_TOKENS" \
# 	--served-model-name "$SERVED_MODEL_NAME" \
# 	--api-key "$API_KEY" \
# 	--benchmark-csv "$BENCHMARK_CSV" \
# 	$(if [ "$ENABLE_PREFIX_CACHING" = "True" ]; then echo "--enable-prefix-caching \"$ENABLE_PREFIX_CACHING\" --prefix-caching-num \"$PREFIX_CACHING_NUM\""; fi)
	#--enable-prefix-caching "$ENABLE_PREFIX_CACHING" \
	#--prefix-caching-num "$PREFIX_CACHING_NUM" 
	


# python benchmark_parallel.py --backend openai \
#       	--host 0.0.0.0 \
# 	--port 9999 \
# 	--tokenizer /work/model/Qwen2___5-0___5B-Instruct  \
# 	--epochs 2 \
# 	--parallel-num 8 \
# 	--prompt-tokens 2048  \
# 	--output-tokens 500 \
# 	--served-model-name qwen \
# 	--api-key '123456' \
# 	--enable-prefix-caching True \
# 	--prefix-caching-num 1024 \
# 	--benchmark-csv test.csv

