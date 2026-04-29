python benchmark_parallel.py --backend openai-chat \
    --host 0.0.0.0 \
	--port 8000 \
	--tokenizer /work/model/Qwen/Qwen2___5-VL-3B-Instruct-AWQ  \
	--epochs 1 \
	--parallel-num 1  \
	--prompt-tokens 512  \
	--output-tokens 512 \
	--multimodal \
	--random-images \
	--served-model-name Qwen2___5-VL-3B-Instruct-AWQ \
	--api-key '123456' \
	--benchmark-csv benchmark_parallel_512.csv

