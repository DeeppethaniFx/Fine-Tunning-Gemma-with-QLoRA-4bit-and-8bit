# Fine Tunning Gemma with QLoRA(4bit & 8bit)

## Introduction to QLoRA
Quantized Low-Rank Adaptation (QLoRA) is a technique that enables fine-tuning large language models efficiently by using quantized weights and low-rank adaptation. This approach significantly reduces memory usage while maintaining model performance, making it suitable for training on consumer-grade GPUs. QLoRA leverages techniques such as 4-bit NormalFloat (NF4) and 8-bit quantization to optimize model storage and computation without sacrificing much accuracy.

## Overview
This repository contains two Jupyter notebooks for fine-tuning the Gemma model using QLoRA with 4-bit and 8-bit quantization. The comparison below highlights the differences in memory usage, model parameters, and efficiency.

## Notebooks
- **Gemma_Finetuning_with_QLoRA_4bit.ipynb**: Implements 4-bit quantization using NF4 (NormalFloat 4-bit) for efficient memory usage.
- **Gemma_Finetuning_with_QLoRA_8bit.ipynb**: Implements 8-bit quantization, offering a balance between precision and memory efficiency.

## Comparison: 4-bit vs 8-bit

| Feature              | 4-bit QLoRA | 8-bit QLoRA |
|----------------------|------------|------------|
| **Quantization Type** | NF4 (4-bit NormalFloat) | 8-bit Integer |
| **Memory Usage**     | Lower (~2x memory savings vs 8-bit) | Higher (uses more memory) |
| **Precision**        | Lower | Higher |
| **Inference Speed**  | Faster due to reduced memory bandwidth | Slightly slower |
| **Training Efficiency** | More efficient on limited GPU memory | Requires more memory but may have better accuracy |
| **Best Use Case**    | When memory is a constraint | When precision is more important |

### Memory Usage Comparison on Google Colab (T4 GPU)
#### 4-bit QLoRA
![4-bit Memory Usage](https://github.com/DeeppethaniFx/Fine-Tunning-Gemma-with-QLoRA-4bit-and-8bit/blob/main/memory_usages_photos/QLoRA_Model_4bit.jpg)

#### 8-bit QLoRA
![8-bit Memory Usage](https://github.com/DeeppethaniFx/Fine-Tunning-Gemma-with-QLoRA-4bit-and-8bit/blob/main/memory_usages_photos/QLoRA_Model_8bit.jpg)

## Dependencies
Both notebooks use the following libraries:
- `bitsandbytes`
- `loralib`
- `peft`
- `accelerate`
- `datasets`
- `transformers`
- `trl`

## Conclusion
- **Use 4-bit QLoRA** if you are constrained by GPU memory and need efficiency.
- **Use 8-bit QLoRA** if you have enough memory and require better model precision.

These notebooks provide a practical implementation of QLoRA for Gemma, helping optimize large models for different hardware capabilities.
