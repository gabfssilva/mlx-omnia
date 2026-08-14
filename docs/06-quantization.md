# 06: Quantization

Storing weights in fewer bits. On a memory-bound decode, this is the single largest lever there is, and it is the one that can silently change what the model says.

## Why it is the main lever

A decode step reads every weight it uses, once, and does almost no arithmetic with each (chapter 07). Time per token is therefore approximately

```
bytes read / memory bandwidth
```

Halving the bytes roughly halves the time. The memory bus sets this limit, so kernel changes cannot recover bytes already read. A 4-bit model reads one quarter of the weight data used by the same model in bf16.

Quantization also decides what *fits*. On a unified-memory machine, the difference between a model that is resident and one that is not is the difference between generating and swapping.

## Group-wise affine quantization

Using one scale for a whole tensor fails because weight magnitudes vary across a matrix. One outlier row can set a scale that wastes most of the range everywhere else.

The fix is to quantize in **groups** along the input dimension. For each contiguous run of `group_size` weights, store:

- `group_size` integers of `bits` bits each,
- a `scale` (fp16/bf16),
- a `bias`: the zero point, i.e. the value that integer 0 decodes to.

Dequantization is `w ≈ q · scale + bias`, per group. With `group_size = 64` and `bits = 4`, the overhead is two 16-bit values per 64 weights: about 0.5 bits per weight, so "4-bit" is really about 4.5.

The integers are **packed into `uint32` words**, eight 4-bit values per word. A packed tensor's last dimension is therefore `input_dims · bits / 32`. `quant/quantization.py` validates this packed shape against the logical one.

`quant/quantization.py` admits `group_size ∈ {32, 64, 128}` and `bits ∈ {2,3,4,5,6,8}` for this format. Smaller groups are more faithful and cost more overhead; that is the whole trade.

## Microscaling formats

Newer formats replace the affine pair with a shared *exponent*, and fix the geometry:

| format | group | bits | scale |
| --- | --- | --- | --- |
| `mxfp4` | 32 | 4 | e8m0 (power of two) |
| `mxfp8` | 32 | 8 | e8m0 |
| `nvfp4` | 16 | 4 | e4m3 (a real float) |

The elements themselves are small floats rather than integers, so the format carries a sign/exponent/mantissa structure inside each group instead of a linear ramp.

This changes two parts of the code. Microscaling formats have **no bias** because the scale carries everything. `biases` is therefore `None` outside affine mode. MLX drops a `None` attribute from the parameter tree, so `load_weights(strict=True)` still matches each mode exactly (`core/layers.py::QuantizedSwitchLinear`).

NVFP4 also gets a separate type from MXFP. Its e4m3 scales and groups of 16 violate assumptions made by MXFP kernels. `quant/quantization.py` represents the formats as three dataclasses (`Affine`, `MXFP`, `NVFP`) with independent validation.

## Method vs format

Worth separating, because they are routinely conflated:

- **Format** is how the bits are laid out: affine/mxfp/nvfp, group size, width. It determines what the kernel does.
- **Method** is how the integers were *chosen*: round-to-nearest, AWQ, GPTQ, or a sensitivity-driven allocator.

The codebase states the separation in a protocol docstring:

> The method (RTN, AWQ, …) is a property of the operation, not of the result: the same format loads identically whichever produced it.

So a better method costs offline time and buys accuracy at the same runtime cost. That is usually the right trade, and it is why the methods exist:

- **RTN**: round each weight to the nearest representable value. Free, and the baseline.
- **AWQ**: some input channels matter much more than others; scale those up before rounding and compensate in the activations, so the rounding error lands where it matters least. Needs a calibration set.
- **GPTQ**: quantize column by column, and after each one adjust the remaining columns to compensate for the error just introduced (using second-order information from calibration activations).

## Mixed precision

A model can mix formats because layers tolerate quantization differently. Extra bits usually pay off in attention projections and the head, which every token reads and which have little redundancy. Wide expert stacks often tolerate narrower formats.

Two mechanisms in this codebase make per-leaf formats possible:

- `core/layers.py::SegmentedQKV`: q, k and v kept as three physical leaves rather than one fused matrix. Concatenating weights requires a common format; *sharing the input* does not. So a checkpoint with q at 4 bits, k at 8 and v dense loads without widening or requantizing anything, and the model splits the concatenated output exactly as it does on the fused path.
- `SegmentedLinear`: the same for projections the checkpoint does not name q/k/v.

The cost is three dispatches instead of one during decode. The fused path remains the default; mixed plans use the segmented path when formats differ.

### Choosing the plan from data

Deciding widths by hand does not scale past a few leaves. `quant/oq.py` measures instead:

```
sensitivity(block) = MSE(float output, quantized output) / mean(float output²)
```

Run a block on calibration data, run it again with its quantizable leaves replaced by quantize-dequantize replicas, and compare the outputs. Both sums accumulate in fp32 over the whole corpus so the element count cancels and the ratio is a corpus-wide mean, not a mean of means.

Two design decisions in that file that are worth copying:

- **The measurement is per block, and the score is reused by every leaf inside it.** Leaves inside one block may still get different widths, but by a *rule on the tensor type* in the allocator: never by three independent measurements. Measuring each leaf separately measures noise as often as signal.
- **Nothing perturbs the model.** Calibration runs the block, swaps in replicas, runs it again and restores the originals; the observation carries both outputs.

## What it costs numerically

Quantization changes the weights and therefore the logits. Unlike the other optimizations in this book, its validation bounds numerical change:

- The comparison is against the same model in full precision on the same inputs, with a measured noise floor: a bf16-vs-bf16 comparison has a floor of its own, and a tolerance that was not measured is a tolerance that will be relaxed later to make a test pass.
- Comparison-side arithmetic runs in fp32 even when the models compared are not.
- Aggregate metrics (perplexity, KL divergence against the full-precision model, top-1 agreement) say more than a single max-difference, because quantization error is distributional.

## Where it lives

- `quant/quantization.py`: the format types, their validation, and the shape arithmetic for packed tensors.
- `quant/awq.py`, `quant/gptq.py`: methods.
- `quant/calibration.py`: running blocks with and without replicas over a corpus.
- `quant/oq.py`, `quant/oqe.py`: sensitivity measurement and the width allocator.
- `core/layers.py`: `QuantizedSwitchLinear`, `QuantizedMultiLinear`, `SegmentedQKV`, `SegmentedLinear`: where a format meets a layer.
- `core/kernels/shared/`: per-format Metal decode helpers, one module per format.
