# PsyMOSAIC-Omni: A Multi-Agent Omni-Modal Framework for Unified Mental Disorder Diagnosis and Multimodal Therapy
<p align="center">
  <a href="https://huggingface.co/Harry-1234/PsyMOSAIC-Omni" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue">
</p>
    
## 👀 PsyMOSAIC-Omni Overview
<p align="center">
    <img src="./assets/method.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="./assets/visualization.png" width="100%" height="100%">
</p>

Recent advances in artificial intelligence have demonstrated great potential for mental disorder diagnosis and personalized therapeutic support. However, existing studies predominantly address diagnostic or therapeutic tasks independently, limiting the effective integration of complementary clinical information across text, audio, and video, while lacking a unified framework that connects mental disorder diagnosis with diverse personalized therapeutic interventions. To address these challenges, we propose PsyMOSAIC-Omni, a multi-agent omni-modal framework for unified mental disorder diagnosis and multimodal therapy, establishing a comprehensive workflow from patient understanding to personalized therapeutic interventions. Specifically, our framework organizes omni-modal patient information into five structured clinical domains and performs mental disorder screening through diagnostic reasoning. Based on the resulting clinical profiles and screening outcomes, specialized agents collaboratively provide personalized multimodal interventions. Furthermore, we develop two task-specialized reinforcement learning methods: PharmaPace-GRPO, which improves medication recommendation through competence-adaptive curriculum learning, and TheraTune-GRPO, which enhances music generation quality through reward-guided policy optimization in continuous latent spaces. Extensive experiments on several benchmarks demonstrate the effectiveness of PsyMOSAIC-Omni in unifying omni-modal mental disorder diagnosis and personalized therapeutic support.

#### 🌟 Contributions in PsyMOSAIC-Omni

1. We develop **PsyMOSAIC-Omni**: A Multi-Agent Omni-Modal Framework for Unified Mental Disorder Diagnosis and Multimodal Therapy. By integrating omni-modal patient understanding with coordinated specialized agents, our framework establishes a comprehensive diagnosis-to-treatment workflow.
2. We annotate three task-specific datasets: **OmniFacet**, which reorganizes multimodal clinical interviews into five structured clinical domains. **SocraDialog**, which provides therapeutic dialogue supervision grounded in patient-specific clinical states and CBT-stage objectives. **PsyRxAtlas**, which supports personalized medication ranking and clinical rationale generation.
3. We develop two task-specialized reinforcement learning methods. **PharmaPace-GRPO** dynamically selects appropriately challenging clinical cases according to the model's evolving competence. **TheraTune-GRPO** employs reward-guided optimization for continuous latent-space music generation.
## 📈 Experimental Results

#### 📍 Results
<p align="center">
    <img src="./assets/model.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="./assets/comparison.png" width="100%" height="100%">
</p>

## ⭐ Training detail and evaluation

### 🔮 Evaluation
```
MODEL_PATH=/path/to/base_model \
DATA_PATH=/path/to/human_behavior_atlas \
ADAPTER=/path/to/checkpoint \
EVAL_SPLIT=test EVAL_PER_DATASET=0 RUN_NAME=hecatepo_test \
bash scripts/run_hecatepo.sh evaluate
```

#### 📖 Prepare
Download the base model [OmniSapiens2.0](https://huggingface.co/HumanBehaviorAtlas/OmniSapiens2.0) and [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) from [huggingface](https://huggingface.co/)

Prepare Hecate-Omni LoRA weights from [huggingface](https://huggingface.co/datasets/Harry-1234/Hecate-Omni)

Prepare human-behavior-atlas data from [huggingface]([https://huggingface.co/datasets/Harry-1234/Hecate-Omni](https://huggingface.co/datasets/HumanBehaviorAtlas/human_behavior_atlas)
### 🔥 Training
```
MODEL_PATH=/path/to/base_model \
DATA_PATH=/path/to/human_behavior_atlas \
RUN_NAME=hecatepo_train \
bash scripts/run_hecatepo.sh train
```
### Examples

https://github.com/user-attachments/assets/000a936d-2834-4798-88f5-3d7d52c58610
```
Question:
How would you describe the emotional state of the speaker in the recording? Choose the most appropriate emotion from:
anger,
disgust,
fear,
happy,
surprise,
neutral,
sad

First reason briefly from the provided evidence. Enclose your reasoning in <think>...</think>. Then provide exactly one final answer in \boxed{...}. For classification, use exactly one of the requested labels. For open questions, give a concise, direct answer.

Hecate-Omni:
<think> The speaker sounds sad because they are expressing that everything is ruined, which is a sad situation. </think> sad \boxed{sad}
```
## 📜 License
Our project is under [MIT License](https://github.com/eeee-sys/Hecate-Omni/blob/main/LICENSE)
