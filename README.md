# **An Integrated AI Agent for CT-Based Quantification, Reasoning, and Communication in Intracerebral Hemorrhage**


<!-- The code of our paper "[**An Integrated AI Agent for CT-Based Quantification, Reasoning, and Communication in Intracerebral Hemorrhage**](https://thank.you)" -->

# Quick Links
+ [Overview](#overview)
+ [Deep Learning-Based Segmentation](#deep-learning-based-segmentation)
+ [Domain-Enhanced Large Language Model for Clinical Decision Support](#domain-enhanced-large-language-model-for-clinical-decision-support)
+ [Digital Avatar Video Synthesis](#digital-avatar-video-synthesis)
+ [License](#license)


# Overview

<div style="text-align: center;">
  <img src="fig/fig.jpg" width="80%">
</div>
The ICH-Agent utilizes a combination of deep learning models for segmentation and reasoning, integrated into a seamless workflow for clinical applications.  

# Key Features
- 🚀 CT-based automated quantification  
- 🧠 Advanced reasoning capabilities  
- 💬 Clinical decision support through a domain-enhanced language model  
- 🎥 Digital avatar video synthesis  

# Deep Learning-Based Segmentation

## nnU-Net V2

To set up nnU-Net V2, you can run the following command:

``` bash
pip install nnunetv2
```

For more usage details, you can refer to [this](https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file#how-to-get-started).

<!-- ## MedNext 

To set up MedNext, you can run the following command:

``` bash
git clone https://github.com/MIC-DKFZ/MedNeXt.git mednext
cd mednext
pip install -e .
```

For more usage details, you can refer to [this](https://github.com/MIC-DKFZ/MedNeXt?tab=readme-ov-file#usage-of-internal-training-pipeline). -->

<!-- ## nnU-Net V2 with ChannelGate

Please replace the `dynamic_network_architectures/unet_decoder.py` in `nnU-Net V2` with `src/unet_decoder.py` -->

#  Domain-Enhanced Large Language Model for Clinical Decision Support

## vLLM

To set up the dependencies, you can run the following command:

``` bash
pip install vLLM
```

vLLM can be deployed as a server that implements the OpenAI API protocol: 

```text-x-trilium-auto
vllm serve /path/to/Qwen3-30B-A3B --enable-reasoning --reasoning-parser deepseek_r1
```

To perform inference with untuned Qwen3-30B-A3B:

``` bash
python /src/infer_api.py
```

To perform inference with domain-enhanced Qwen3-30B-A3B:

```
python /src/infer_api_domain.py
```

For more usage details, you can refer to [this](https://docs.vllm.ai/en/latest/).

# Digital Avatar Video Synthesis

## Duix.Avatar (formerly Heygem)

Install the server:

```
cd /src/deploy_heygem
docker-compose -f docker-compose-linux.yml up -d
```

Install the client:

1. Directly download the Linux version of the [officially built installation package](https://github.com/GuijiAI/HeyGem.ai/releases).
2. Double click `HeyGem-x.x.x.AppImage` to launch it. No installation is required.

For more usage details, you can refer to [this](https://github.com/duixcom/Duix.Heygem?tab=readme-ov-file#3-how-to-run-locally).

## Example
https://github.com/user-attachments/assets/09b6615a-c77b-42bc-b92f-4ad3f84f1777

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
