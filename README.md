# Q-Infer

## Getting Started

- [Installation](#setup-and-installation)
- [Model Weights](#model-weights)
- [Inference](#inference)

## Setup and Installation

### Pre-requisites

Requires the following dependencies:

- CMake (3.17+)
- Python (3.8+) and pip (19.3+), for converting model weights and automatic FFN offloading

```bash
cd Q-infer
pip install -r requirements.txt # install Python helpers' dependencies
```
### Build

Using `CMake`(3.17+):
* Build on NVIDIA GPU:
```bash
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
```
* Build on NPU:
```bash
cmake -S . -B build -DLLAMA_CANN=ON
cmake --build build --config Release
```

## Model Weights

Q-Infer based on PowerInfer models, which are stored in a special format called *PowerInfer GGUF* based on GGUF format, consisting of both LLM weights and predictor weights.

### Download PowerInfer GGUF via Hugging Face

You can obtain PowerInfer GGUF weights at `*.powerinfer.gguf` as well as profiled model activation statistics for 'hot'-neuron offloading from each Hugging Face repo below.

| Base Model            | PowerInfer GGUF                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| LLaMA(ReLU)-2-7B      | [PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF)     |
| LLaMA(ReLU)-2-13B     | [PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF)   |
| Falcon(ReLU)-40B      | [PowerInfer/ReluFalcon-40B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluFalcon-40B-PowerInfer-GGUF) |
| LLaMA(ReLU)-2-70B     | [PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF)   |
| ProSparse-LLaMA-2-7B  | [PowerInfer/ProSparse-LLaMA-2-7B-GGUF](https://huggingface.co/PowerInfer/prosparse-llama-2-7b-gguf)           |
| ProSparse-LLaMA-2-13B | [PowerInfer/ProSparse-LLaMA-2-13B-GGUF](https://huggingface.co/PowerInfer/prosparse-llama-2-13b-gguf)         |
| Bamboo-base-7B 🌟      | [PowerInfer/Bamboo-base-v0.1-gguf](https://huggingface.co/PowerInfer/Bamboo-base-v0.1-gguf)                   |
| Bamboo-DPO-7B 🌟       | [PowerInfer/Bamboo-DPO-v0.1-gguf](https://huggingface.co/PowerInfer/Bamboo-DPO-v0.1-gguf)                     |

We recommend using [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/guides/cli) to download the whole model repo. For example, the following command will download [PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF) into the `./ReluLLaMA-7B` directory.

```shell
huggingface-cli download --resume-download --local-dir ReluLLaMA-7B --local-dir-use-symlinks False PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF
```

As such, PowerInfer can automatically make use of the following directory structure for feature-complete model offloading:
```
.
├── *.powerinfer.gguf (Unquantized PowerInfer model)
├── *.q4.powerinfer.gguf (INT4 quantized PowerInfer model, if available)
├── activation (Profiled activation statistics for fine-grained FFN offloading)
│   ├── activation_x.pt (Profiled activation statistics for layer x)
│   └── ...
├── *.[q4].powerinfer.gguf.generated.gpuidx (Generated GPU index at runtime for corresponding model)
```

### Convert from Original Model Weights + Predictor Weights

Hugging Face limits single model weight to 50GiB. For unquantized models >= 40B, you can convert PowerInfer GGUF from the original model weights and predictor weights obtained from Hugging Face.

| Base Model            | Original Model                                                                            | Predictor                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| LLaMA(ReLU)-2-7B      | [SparseLLM/ReluLLaMA-7B](https://huggingface.co/SparseLLM/ReluLLaMA-7B)                   | [PowerInfer/ReluLLaMA-7B-Predictor](https://huggingface.co/PowerInfer/ReluLLaMA-7B-Predictor)                   |
| LLaMA(ReLU)-2-13B     | [SparseLLM/ReluLLaMA-13B](https://huggingface.co/SparseLLM/ReluLLaMA-13B)                 | [PowerInfer/ReluLLaMA-13B-Predictor](https://huggingface.co/PowerInfer/ReluLLaMA-13B-Predictor)                 |
| Falcon(ReLU)-40B      | [SparseLLM/ReluFalcon-40B](https://huggingface.co/SparseLLM/ReluFalcon-40B)               | [PowerInfer/ReluFalcon-40B-Predictor](https://huggingface.co/PowerInfer/ReluFalcon-40B-Predictor)               |
| LLaMA(ReLU)-2-70B     | [SparseLLM/ReluLLaMA-70B](https://huggingface.co/SparseLLM/ReluLLaMA-70B)                 | [PowerInfer/ReluLLaMA-70B-Predictor](https://huggingface.co/PowerInfer/ReluLLaMA-70B-Predictor)                 |
| ProSparse-LLaMA-2-7B  | [SparseLLM/ProSparse-LLaMA-2-7B](https://huggingface.co/SparseLLM/prosparse-llama-2-7b)   | [PowerInfer/ProSparse-LLaMA-2-7B-Predictor](https://huggingface.co/PowerInfer/prosparse-llama-2-7b-predictor)   |
| ProSparse-LLaMA-2-13B | [SparseLLM/ProSparse-LLaMA-2-13B](https://huggingface.co/SparseLLM/prosparse-llama-2-13b) | [PowerInfer/ProSparse-LLaMA-2-13B-Predictor](https://huggingface.co/PowerInfer/prosparse-llama-2-13b-predictor) |
| Bamboo-base-7B 🌟      | [PowerInfer/Bamboo-base-v0.1](https://huggingface.co/PowerInfer/Bamboo-base-v0_1)         | [PowerInfer/Bamboo-base-v0.1-predictor](https://huggingface.co/PowerInfer/Bamboo-base-v0.1-predictor)           |
| Bamboo-DPO-7B 🌟       | [PowerInfer/Bamboo-DPO-v0.1](https://huggingface.co/PowerInfer/Bamboo-DPO-v0_1)           | [PowerInfer/Bamboo-DPO-v0.1-predictor](https://huggingface.co/PowerInfer/Bamboo-DPO-v0.1-predictor)             |

You can use the following command to convert the original model weights and predictor weights to PowerInfer GGUF:
```bash
# make sure that you have done `pip install -r requirements.txt`
python convert.py --outfile /PATH/TO/POWERINFER/GGUF/REPO/MODELNAME.powerinfer.gguf /PATH/TO/ORIGINAL/MODEL /PATH/TO/PREDICTOR
# python convert.py --outfile ./ReluLLaMA-70B-PowerInfer-GGUF/llama-70b-relu.powerinfer.gguf ./SparseLLM/ReluLLaMA-70B ./PowerInfer/ReluLLaMA-70B-Predictor
```
For the same reason, we suggest keeping the same directory structure as PowerInfer GGUF repos after conversion.

<details>

<summary>Convert Original models into dense GGUF models(compatible with llama.cpp)</summary>

```bash
python convert-dense.py --outfile /PATH/TO/DENSE/GGUF/REPO/MODELNAME.gguf /PATH/TO/ORIGINAL/MODEL
# python convert-dense.py --outfile ./Bamboo-DPO-v0.1-gguf/bamboo-7b-dpo-v0.1.gguf --outtype f16 ./Bamboo-DPO-v0.1
```

Please note that the generated dense GGUF models might not work properly with llama.cpp, as we have altered activation functions (for ReluLLaMA and Prosparse models), or the model architecture (for Bamboo models). The dense GGUF models generated by convert-dense.py can be used for PowerInfer in dense inference mode, but might not work properly with llama.cpp.

</details>

## Inference

For CPU-GPU hybrid inference with all available VRAM, you can use the following instructions to run:
```bash
./build/bin/main -m /PATH/TO/MODEL -n $output_token_count -t $thread_num -p $prompt
# e.g.: ./build/bin/main -m ./ReluFalcon-40B-PowerInfer-GGUF/falcon-40b-relu.q4.powerinfer.gguf -n 128 -t 8 -p "Once upon a time"
# For Windows: .\build\bin\Release\main.exe -m .\ReluFalcon-40B-PowerInfer-GGUF\falcon-40b-relu.q4.powerinfer.gguf -n 128 -t 8 -p "Once upon a time"
```

If you want to limit the VRAM usage of GPU:
```bash
./build/bin/main -m /PATH/TO/MODEL -n $output_token_count -t $thread_num -p $prompt --vram-budget $vram_gb
# e.g.: ./build/bin/main -m ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf -n 128 -t 8 -p "Once upon a time" --vram-budget 8
# For Windows: .\build\bin\Release\main.exe -m .\ReluLLaMA-7B-PowerInfer-GGUF\llama-7b-relu.powerinfer.gguf -n 128 -t 8 -p "Once upon a time" --vram-budget 8
```
Under CPU-GPU hybrid inference, PowerInfer will automatically offload all dense activation blocks to GPU, then split FFN and offload to GPU if possible.

<details>
<summary>Dense inference mode (limited support)</summary>

If you want to run PowerInfer to infer with the dense variants of the PowerInfer model family, you can use similarly as llama.cpp does:

```bash
./build/bin/main -m /PATH/TO/DENSE/MODEL -n $output_token_count -t $thread_num -p $prompt -ngl $num_gpu_layers
# e.g.: ./build/bin/main -m ./Bamboo-base-v0.1-gguf/bamboo-7b-v0.1.gguf -n 128 -t 8 -p "Once upon a time" -ngl 12
```

So is the case for other `examples/` like `server` and `batched_generation`. Please note that the dense inference mode is not a "compatible mode" for all models. We have altered activation functions (for ReluLLaMA and Prosparse models) in this mode to match with our model family. 

</details>

## Serving, Perplexity Evaluation, and more applications

PowerInfer supports serving and batched generation with the same instructions as llama.cpp. Generally, you can use the same command as llama.cpp, except for `-ngl` argument which has been replaced by `--vram-budget` for PowerInfer. Please refer to the detailed instructions in each `examples/` directory. For example:

- [Serving](./examples/server/README.md)
- [Perplexity Evaluation](./examples/perplexity/README.md)
- [Batched Generation](./examples/batched/README.md)

## Quantization

PowerInfer has optimized quantization support for INT4(`Q4_0`) models. You can use the following instructions to quantize PowerInfer GGUF model:
```bash
./build/bin/quantize /PATH/TO/MODEL /PATH/TO/OUTPUT/QUANTIZED/MODEL Q4_0
# e.g.: ./build/bin/quantize ./ReluFalcon-40B-PowerInfer-GGUF/falcon-40b-relu.powerinfer.gguf ./ReluFalcon-40B-PowerInfer-GGUF/falcon-40b-relu.q4.powerinfer.gguf Q4_0
# For Windows: .\build\bin\Release\quantize.exe .\ReluFalcon-40B-PowerInfer-GGUF\falcon-40b-relu.powerinfer.gguf .\ReluFalcon-40B-PowerInfer-GGUF\falcon-40b-relu.q4.powerinfer.gguf Q4_0
```
Then you can use the quantized model for inference with the same instructions as above.

## FAQs
1. What if I encountered `CUDA_ERROR_OUT_OF_MEMORY`?
   - You can try to run with `--reset-gpu-index` argument to rebuild the GPU index for this model to avoid any stale cache.
   - Due to our current implementation, model offloading might not be as accurate as expected. You can try with `--vram-budget` with a slightly lower value.

2. Why is there a noticeable downgrade in the performance metrics of our current ReLU model, particularly the 70B model?
   - In contrast to the typical requirement of around 2T tokens for LLM training, our model's fine-tuning was conducted with only 5B tokens. This insufficient retraining has resulted in the model's inability to regain its original performance. We are actively working on updating to a more capable model, so please stay tuned.
