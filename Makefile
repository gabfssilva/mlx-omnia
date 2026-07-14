# The -Package scheme is the auto-generated one that carries the test action now that
# the package has two products.
SCHEME := Sideros-Package
DESTINATION := platform=macOS,arch=arm64
XCODEBUILD := xcodebuild -scheme $(SCHEME) -destination '$(DESTINATION)'

# MLX routes float32 matmul on Metal through a reduced-precision path by default
# (the M5 Neural Accelerators), which costs ~3 decimal digits. Tests compare against
# torch and mlx-lm in float32, so they need the exact kernels. TEST_RUNNER_ is how
# xcodebuild forwards a variable into the test process.
export TEST_RUNNER_MLX_ENABLE_TF32 := 0

# swift build/test cannot compile mlx-swift's Metal shaders, so the test binary
# aborts at runtime with a missing default.metallib. Everything goes through xcodebuild.
.PHONY: build test test-ci test-api serve weights fixtures bench clean

SERVE_DERIVED := .build/serve
SERVE_BIN := $(SERVE_DERIVED)/Build/Products/Release/sideros-serve
SERVE_BUILD := xcodebuild -quiet -scheme sideros-serve -destination '$(DESTINATION)' \
	-configuration Release -derivedDataPath $(SERVE_DERIVED) build
API_MODEL ?= mlx-community/Qwen2.5-0.5B-Instruct-4bit
API_NAME ?= qwen

serve:
	$(SERVE_BUILD)
	$(SERVE_BIN) $(API_MODEL) --name $(API_NAME)

# The dialects are verified by the SDKs that have to talk to them, not by our own JSON.
test-api:
	$(SERVE_BUILD)
	SIDEROS_SERVE=$(abspath $(SERVE_BIN)) SIDEROS_MODEL=$(API_MODEL) \
		uv run --with pytest --with httpx --with openai --with anthropic --with google-genai \
		--with litellm --no-project pytest reference/test_api.py -q

BENCH_DERIVED := .build/bench
BENCH_BIN := $(BENCH_DERIVED)/Build/Products/Release/sideros-bench
BENCH_PY := uv run --with mlx-lm --no-project reference/bench_mlxlm.py
BENCH_VLM := HF_HUB_OFFLINE=1 uv run --python 3.12 \
	--with 'mlx==0.31.1' --with 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm' \
	--with requests --no-project reference/bench_mlxvlm.py
BENCH_IMAGE := Tests/SiderosTests/Fixtures/qwen3_5_vision.png

# TF32 stays at its default here: bench measures the fast path, parity tests the exact one.
bench:
	xcodebuild -scheme sideros-bench -destination '$(DESTINATION)' -configuration Release \
		-derivedDataPath $(BENCH_DERIVED) build
	$(BENCH_BIN) Qwen/Qwen2.5-0.5B --label qwen2-bf16
	$(BENCH_BIN) Qwen/Qwen2.5-0.5B --quantize --label qwen2-q4-onload
	$(BENCH_BIN) mlx-community/Qwen2.5-0.5B-Instruct-4bit --label qwen2-q4-ckpt
	$(BENCH_BIN) openai-community/gpt2 --label gpt2-fp32
	$(BENCH_BIN) openai-community/gpt2 --quantize --label gpt2-q4
	$(BENCH_BIN) google/gemma-3-270m --label gemma3-bf16
	$(BENCH_BIN) Qwen/Qwen3-14B --label qwen3-14b-bf16
	$(BENCH_BIN) Qwen/Qwen3-14B --quantize --label qwen3-14b-q4
	$(BENCH_BIN) Qwen/Qwen3-14B --draft mlx-community/Qwen3-4B-4bit --label qwen3-14b-spec
	$(BENCH_BIN) LiquidAI/LFM2.5-8B-A1B --label lfm25-8b-a1b-bf16
	$(BENCH_BIN) mlx-community/Qwen3.6-27B-6bit --label qwen36-27b-6bit
	$(BENCH_BIN) mlx-community/Qwen3.6-27B-6bit --image $(BENCH_IMAGE) --label qwen36-27b-image
	$(BENCH_BIN) mlx-community/Qwen3-30B-A3B-4bit --label qwen3-30b-a3b-q4
	$(BENCH_BIN) Qwen/Qwen3.5-0.8B --label qwen35-0.8b-bf16
	$(BENCH_BIN) mlx-community/Qwen3.6-35B-A3B-6bit --label qwen36-35b-a3b-6bit
	$(BENCH_BIN) mlx-community/Qwen3.6-35B-A3B-4bit --label qwen36-35b-a3b-4bit
	$(BENCH_PY) Qwen/Qwen2.5-0.5B
	$(BENCH_PY) mlx-community/Qwen2.5-0.5B-Instruct-4bit
	$(BENCH_PY) Qwen/Qwen3-14B
	$(BENCH_PY) LiquidAI/LFM2.5-8B-A1B
	$(BENCH_PY) mlx-community/Qwen3.6-27B-6bit
	$(BENCH_VLM) ~/.omlx/models/Qwen3.6-27B-6bit $(BENCH_IMAGE)
	$(BENCH_PY) mlx-community/Qwen3-30B-A3B-4bit
	$(BENCH_PY) Qwen/Qwen3.5-0.8B
	$(BENCH_PY) mlx-community/Qwen3.6-35B-A3B-6bit
	$(BENCH_PY) mlx-community/Qwen3.6-35B-A3B-4bit

weights:
	hf download openai-community/gpt2 model.safetensors config.json vocab.json merges.txt
	hf download Qwen/Qwen2.5-0.5B model.safetensors config.json tokenizer.json
	hf download Qwen/Qwen3-0.6B model.safetensors config.json tokenizer.json
	hf download Qwen/Qwen3-14B --include 'model*' 'config.json' 'tokenizer.json'
	hf download mlx-community/Qwen3-0.6B-4bit
	hf download mlx-community/Qwen3-4B-4bit
	hf download google/gemma-3-270m model.safetensors config.json tokenizer.json tokenizer_config.json
	hf download mlx-community/Qwen2.5-0.5B-Instruct-4bit model.safetensors config.json tokenizer.json
	hf download LiquidAI/LFM2.5-8B-A1B model.safetensors config.json tokenizer.json tokenizer_config.json
	hf download Qwen/Qwen3.5-0.8B --include 'model*' 'config.json' 'tokenizer.json' 'preprocessor_config.json'
	hf download mlx-community/Qwen3-30B-A3B-4bit
	hf download mlx-community/Qwen3.6-35B-A3B-4bit

fixtures:
	uv run --with transformers --with safetensors --no-project reference/gen_tokenizer_fixture.py
	uv run --with transformers --no-project reference/gen_bpe_tokenizer_fixture.py
	uv run --with transformers --with jinja2 --no-project reference/gen_chat_template_fixture.py
	uv run --with transformers --with torch --with safetensors --no-project reference/gen_gpt2_fixture.py
	uv run --with transformers --with torch --with safetensors --no-project reference/gen_qwen2_fixture.py
	uv run --with transformers --with torch --with safetensors --no-project reference/gen_qwen3_fixture.py
	uv run --with transformers --with torch --with safetensors --no-project reference/gen_gemma3_fixture.py
	MLX_ENABLE_TF32=0 uv run --with mlx-lm --with safetensors --no-project reference/gen_mlxlm_fixture.py
	MLX_ENABLE_TF32=0 uv run --with mlx-lm --with safetensors --no-project reference/gen_mlxlm_qwen2_fixture.py
	uv run --with mlx --with huggingface_hub --no-project reference/gen_quantize_fixture.py
	MLX_ENABLE_TF32=0 uv run --with mlx-lm --with safetensors --no-project reference/gen_mlxlm_q4_fixture.py
	uv run --with transformers --with torch --with safetensors --no-project reference/gen_lfm2moe_fixture.py
	uv run --with transformers --with torch --with safetensors --no-project reference/gen_qwen3_5_fixture.py
	uv run --with transformers --with torch --with torchvision --with pillow --with safetensors --no-project reference/gen_qwen3_5_vision_fixture.py
	MLX_ENABLE_TF32=0 uv run --with 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm' --with safetensors --no-project reference/gen_qwen3_moe_fixture.py
	MLX_ENABLE_TF32=0 uv run --with 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm' --with safetensors --no-project reference/gen_qwen3_5_moe_fixture.py
	MLX_ENABLE_TF32=0 uv run --python 3.12 --with 'mlx==0.31.1' \
		--with 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm' --with safetensors \
		--no-project reference/gen_qwen3_5_vision_mlxvlm.py

build:
	$(XCODEBUILD) build

# Serial until mlx-swift vendors mlx >= 968d264f (2026-06-10): before it, the tracing
# state is a plain static, so a test tracing a `compile` races any test building ops
# and the suite dies on a libc++ abort. No parallelism, no race — inference itself is
# single-threaded.
#
# Serial is also what keeps the suite inside RAM: every test loads its own checkpoint, and
# in parallel the big ones (30B-A3B 4-bit is 17GB, LFM2.5-8B bf16 ~16GB) are resident at
# once. Paging is what makes a 1-minute suite take 7.
#
# EXTRA is how a single test still runs through here: the parentheses are required.
#   make test EXTRA="-only-testing:'SiderosTests/qwen3MoeLogitsMatchMLXLM()'"
test:
	$(XCODEBUILD) -parallel-testing-enabled NO $(EXTRA) test

# CI has no plugin trust store, so mlx-swift's CudaBuild plugin must be waved through.
test-ci:
	$(XCODEBUILD) -skipPackagePluginValidation -parallel-testing-enabled NO test

clean:
	$(XCODEBUILD) clean
	rm -rf .build
