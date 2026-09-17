# Explorer Survey Arch Task

You are teamwork_preview_explorer_survey_2.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

Your task:
Investigate the runtime environment (Python version, PyTorch, Transformers, hardware acceleration MPS/CUDA/CPU, memory limitations), the Qwen/Qwen1.5-MoE-A2.7B model architecture (number of layers, number of experts, top-k gating, router logits structure, hidden state dimensions, Layer N selection), data streaming mechanics for a 100k-token corpus without OOM, and strictly isolated 15-20% held-out calibration split.
Produce a comprehensive analysis.md and handoff.md in your working directory.
Reference /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md.

## 2026-09-17T07:24:32Z
Investigate the technical environment and model architecture:
1. Local environment: Python version, PyTorch version, Transformers version, accelerate/datasets, available compute devices (MPS on Mac, CPU, CUDA if any), RAM constraints, and disk space.
2. Qwen/Qwen1.5-MoE-A2.7B architecture: Inspect model configuration, total layers, hidden dimension (d_model), number of routed experts, top-k routed experts, shared experts (if any), routing mechanism (router logits, softmax vs sigmoid top-k, gating output tensors). Determine suitable Layer N for tap (e.g. intermediate layer such as layer 2, 3, or 4 prior to deep layers 5-24).
3. Streaming & Memory management: How to stream 100k tokens through the model without OOM on Mac/local resources. Batch sizing, torch.no_grad, activation offloading/saving to disk (memmap / safetensors / zarr / torch save in chunks), memory leak prevention.
4. Clean train/calibration split: 15-20% held-out strictly isolated split mechanics.
