# Few-Shot Audio-Visual Deepfake Detection via Regret-Matched Synchrony Ensembles

[![BMVC 2026](https://img.shields.io/badge/BMVC-2026-blue.svg)](https://bmvc2026.org/)
[![RVS-SE Workshop](https://img.shields.io/badge/Workshop-RVS--SE-orange.svg)](https://rvsse.github.io/)

## Overview

This repository contains the work **“Few-Shot Audio-Visual Deepfake Detection via Regret-Matched Synchrony Ensembles”**, accepted at the **Workshop on Robust Vision Systems in Synthetic Environments (RVS-SE), in conjunction with BMVC 2026**.

The paper presents a unified few-shot framework for detecting audio, visual, and audio-visual deepfakes when only a handful of labeled examples are available. The framework combines frozen CLIP and Wav2Vec2 foundation encoders with lightweight residual adapters, a meta-learned regret-matched synchrony ensemble, gated cross-modal fusion, modality dropout, prototype-based episodic classification, and a Task Consistency Regularizer.

The official RVS-SE website lists this paper as **Paper #14**.

## Key Contributions

- **Unified few-shot detection:** Handles audio-only, video-only, and audio-visual deepfake detection in one episodic architecture.
- **Parameter-efficient adaptation:** Frozen CLIP and Wav2Vec2 backbones are adapted through lightweight residual adapters.
- **Regret-matched synchrony ensemble:** Multiple synchrony estimators are combined through online regret-based weighting.
- **Four-way modality attribution:** Distinguishes Real Video–Real Audio, Fake Video–Fake Audio, Fake Video–Real Audio, and Real Video–Fake Audio.
- **Gated cross-modal fusion:** Allows each modality to selectively incorporate information from the other.
- **Modality dropout:** Improves robustness to missing and partially manipulated modalities.
- **Task Consistency Regularizer:** Stabilizes class prototypes under extreme label scarcity.
- **Identity-disjoint evaluation:** Support and query identities are separated.
- **Cross-dataset evaluation:** Evaluated between FakeAVCeleb and PolyGlotFake.

## Model Architecture

The framework contains four major components:

1. Parameter-Efficient Bimodal Encoding
2. Meta-Learned Synchrony Ensemble
3. Gated Cross-Modal Fusion with Modality Dropout
4. Prototype-Based Episodic Classification with Task Consistency Regularization

### Parameter-Efficient Bimodal Encoding

**Visual branch:** Uniformly sampled video frames are arranged into a 2 × (Nf/2) collage and resized to 224 × 224. The collage is encoded using frozen **CLIP ViT-B/16**.

**Audio branch:** Audio is resampled to 16 kHz, converted to mono, cropped/padded to a fixed duration, and encoded using frozen **Wav2Vec2**.

### Residual Adapters

Both foundation encoders remain frozen. Lightweight residual adapters perform adaptation:

    Ada(h) = h + W2 φ(W1 LN(h))

The up-projection W2 is zero-initialized, making the adapter an identity mapping at initialization.

## Meta-Learned Synchrony Ensemble

The model constructs an interaction descriptor:

    u = [v ; a ; v ⊙ a ; |v-a| ; cos(v,a)]

Multiple lightweight MLP synchrony estimators predict audio-visual coherence scores.

Instead of a conventional learned softmax gate, the framework uses **online regret matching**. Estimators that outperform the recent ensemble average receive larger weights. Entropy and load-balancing terms prevent collapse onto a single estimator.

## Gated Cross-Modal Fusion

The visual and audio representations selectively incorporate information from each other through learned scalar gates. The gated representations are processed by a two-layer MLP with layer normalization.

This mechanism reduces the influence of unreliable modalities while preserving complementary cross-modal information.

## Modality Dropout

During training, either modality may be randomly suppressed. This encourages complementary representations and improves robustness to:

- Audio-only manipulation
- Video-only manipulation
- Missing audio
- Missing video
- Partially manipulated samples

## Four-Way Classification

| Class | Video | Audio |
|---|---|---|
| RV-RA | Real | Real |
| FV-FA | Fake | Fake |
| FV-RA | Fake | Real |
| RV-FA | Real | Fake |

The formulation requires the model to identify not only whether content is manipulated, but also which modality is responsible.

For unimodal inference, the missing modality is replaced by a learnable null embedding and cross-modal components are disabled.

## Few-Shot Episodic Learning

The model follows a K-way M-shot episodic protocol:

1. Sample a subset of classes.
2. Select M labeled support examples per class.
3. Sample a separate query set.
4. Keep support/query identities mutually disjoint.
5. Adapt trainable modules using only the support set.
6. Evaluate on the query set.

This enables adaptation to manipulation categories using only a small number of labeled examples.

## Prototype-Based Classification

Class prototypes are computed from support embeddings and query samples are classified according to their distance to the prototypes.

The prototype formulation also allows new manipulation classes to be incorporated by computing their prototypes without gradient updates.

## Task Consistency Regularizer

To reduce instability caused by very small support sets, multiple support sub-tasks are created and their class prototypes are encouraged to remain close to a consensus prototype.

TCR is omitted for the 1-shot setting.

## Training Objective

The complete objective is:

    L = Lcls + λcl Lnce + λsync Lsync + λproto Lproto + λtcr Ltcr

where:

- **Lcls:** Focal classification loss
- **Lnce:** Class-conditioned cross-modal InfoNCE
- **Lsync:** Synchrony ensemble objective
- **Lproto:** Prototypical loss
- **Ltcr:** Task Consistency Regularizer

## Datasets

### FakeAVCeleb

Approximately 25K synchronized face-speech samples covering:

- Real Video – Real Audio
- Real Video – Fake Audio
- Fake Video – Real Audio
- Fake Video – Fake Audio

### PolyGlotFake

Approximately 14.2K multilingual audio-visual samples across seven languages, including facial manipulation, text-to-speech, voice cloning, and lip-synchronization manipulations.

### ASVspoof 2019 LA

96,617 speech samples for audio deepfake detection, including text-to-speech and voice-conversion attacks.

### FaceForensics++

Used for video-only deepfake detection with original videos as real samples and manipulated videos under the c23 compression setting.

## Experimental Protocol

For each M-shot setting:

- M samples form the support set.
- Remaining samples form the query set.
- 20 independent episodes are evaluated.
- Support and query identities are disjoint.
- Results are reported as mean ± 95% confidence interval.

Metrics include Accuracy, AUC, F1, and EER for audio detection.

## Few-Shot Audio-Visual Results

### FakeAVCeleb — 4-Way Classification

| Method | M=1 ACC | M=5 ACC | M=10 ACC | M=10 AUC |
|---|---:|---:|---:|---:|
| SSL-ProtoNet | 20.2 | 21.5 | 22.6 | 49.8 |
| Linear Probe | 19.3 | 20.2 | 21.6 | 48.9 |
| Full Fine-Tuning | 23.4 | 24.7 | 25.2 | 50.5 |
| AVoiD-DF | 22.6 | 23.9 | 24.7 | 48.8 |
| AVFakeNet | 23.9 | 25.2 | 26.6 | 51.2 |
| **Ours** | **28.4** | **31.5** | **32.8** | **65.5** |

At 10 shots on FakeAVCeleb, the method achieves **32.8% accuracy, 29.2% macro-F1, and 65.5% AUC**.

### PolyGlotFake — 2-Way Classification

At 10 shots, the method achieves:

- **63.1% Accuracy**
- **58.4% macro-F1**
- **69.8% AUC**

## Unimodal Few-Shot Detection

### Audio — ASVspoof 2019 LA

| Method | 1-shot EER | 5-shot EER | 10-shot EER |
|---|---:|---:|---:|
| AASIST | 8.26 | 8.06 | 7.88 |
| AASIST2 | 6.38 | 6.12 | 5.92 |
| W2V2 Linear Probe | 9.48 | 9.22 | 9.04 |
| **Ours** | **4.72** | **4.24** | **4.06** |

### Video — FaceForensics++

| Method | 1-shot ACC | 5-shot ACC | 10-shot ACC |
|---|---:|---:|---:|
| DiffFake | 51.7 | 53.6 | 57.5 |
| FSBI | 53.4 | 57.8 | 61.8 |
| CLIP Linear Probe | 42.3 | 45.6 | 48.9 |
| **Ours** | **63.2** | **68.6** | **72.5** |

At 10 shots, the video mode reaches **72.5% accuracy and 84.4% AUC**.

## Cross-Dataset Generalization

At 5 shots:

| Adaptation → Evaluation | ACC | AUC |
|---|---:|---:|
| FakeAVCeleb → PolyGlotFake | 57.8 | 65.2 |
| PolyGlotFake → FakeAVCeleb | 53.7 | 62.9 |

## Ablation Study

Episode-matched ablations on FakeAVCeleb show the following macro-F1 at 5 shots:

| Variant | Macro-F1 | Drop |
|---|---:|---:|
| Full Model | 27.9 | — |
| w/o Synchrony Loss | 23.2 | 4.7 |
| w/o Gated Fusion | 21.7 | 6.2 |
| w/o Modality Dropout | 22.6 | 5.3 |
| w/o Class-Conditional InfoNCE | 22.6 | 5.3 |
| w/o Prototypical Loss | 22.2 | 5.7 |
| w/o TCR | 22.8 | 5.1 |
| w/o Residual Adapters | 19.8 | 8.1 |

### Synchrony Aggregation

| Strategy | Macro-F1 at M=5 |
|---|---:|
| Full Regret Matching | 27.9 |
| Single Expert | 19.2 |
| Uniform Average | 19.2 |
| Learned Softmax Gate | 20.0 |
| w/o Load-Balance / Entropy Bonus | 20.8 |

### Visual Input

| Visual Input | Macro-F1 at M=5 |
|---|---:|
| 8-frame Collage | 27.9 |
| Single Frame | 15.6 |
| Per-frame + Mean Pooling | 16.4 |

## Qualitative Explainability

Gradient-based saliency maps are evaluated for both modalities.

**Visual:** Saliency concentrates on facial regions such as the mouth, eyes, and facial contours, while background regions receive little attribution.

**Audio:** Saliency exhibits localized peaks around speech onsets and spectral transitions, with fake-audio samples showing concentrated temporal regions that may correspond to vocoder artifacts.

## Parameter Efficiency

| Method | Trainable Params | % of Full FT | Latency | Adaptation / Episode |
|---|---:|---:|---:|---:|
| Full Fine-Tuning | 180.6M | 100% | 44 ms | 96 s |
| AVoiD-DF | 112.4M | 62.2% | 57 ms | 74 s |
| AVFakeNet | 134.8M | 74.6% | 83 ms | 92 s |
| **Ours** | **8.6M** | **4.8%** | **46 ms** | **31 s** |

On an NVIDIA T4, the method updates only **8.6M parameters (4.8% of full fine-tuning)**, with **46 ms per-clip latency** and **31 s adaptation time per episode**.

## Main Findings

- Few-shot audio-visual deepfake detection is considerably more difficult than unimodal detection.
- Frozen foundation encoders plus lightweight adapters provide parameter-efficient adaptation.
- Regret-matched synchrony improves episode-wise selection of complementary synchrony estimators.
- Gated fusion regulates cross-modal information flow.
- Modality dropout supports partially manipulated and missing-modality inputs.
- Prototype learning and TCR improve few-shot representation stability.
- The same framework supports audio-only, video-only, and audio-visual inference.
- Cross-dataset evaluation demonstrates transfer between FakeAVCeleb and PolyGlotFake.
- The strongest results are obtained in the unimodal settings, while reliable four-way modality attribution remains challenging.

## Limitations

The paper identifies several limitations:

1. Absolute four-way and partial-manipulation performance remains modest under 10-shot supervision.
2. Robustness to recompression, resizing, frame-rate changes, and other realistic social-media degradation has not been evaluated.
3. The synchrony supervision treats jointly manipulated FV-FA content as unsynchronized even though such content can remain temporally synchronized.
4. The frame-collage representation captures coarse inter-frame structure rather than continuous motion.

## Future Directions

- Fine-grained temporal alignment
- Real-world media degradation evaluation
- Stronger few-shot deepfake baselines
- Open-set and continual detection
- More diverse and localized manipulations
- Weaker synchronization conditions
- Improved uncertainty modeling
- Larger-scale multimodal pretraining

## Repository Structure

The exact repository structure and runnable training scripts can be added when the implementation is released.

Suggested organization:

    .
    ├── README.md
    ├── configs/
    ├── datasets/
    ├── models/
    │   ├── adapters/
    │   ├── synchrony/
    │   ├── fusion/
    │   └── prototypes/
    ├── losses/
    ├── training/
    ├── evaluation/
    └── visualization/

## Getting Started

Implementation and reproducibility instructions should be added when the source code and pretrained models are released.

The experiments use CLIP ViT-B/16, Wav2Vec2, FakeAVCeleb, PolyGlotFake, ASVspoof 2019 LA, and FaceForensics++.

Dataset access and licensing requirements should be followed according to the respective dataset providers.

## Citation

    

## Authors

**Saptarshi Pani** — Jadavpur University, Kolkata, India  
**Rajdeep Pal** — St Thomas’ College of Engineering and Technology, Kolkata, India  
**Jotiraditya Banerjee** — Jadavpur University, Kolkata, India  
**Sergei Romanov** — Saint Petersburg Electrotechnical University "LETI", Russia  
**Dmitrii Kaplun** — Saint Petersburg Electrotechnical University "LETI", Russia  
**Ram Sarkar** — Jadavpur University, Kolkata, India

All authors contributed equally to this work.

## Workshop

**Workshop on Robust Vision Systems in Synthetic Environments (RVS-SE)**  
**In conjunction with BMVC 2026**

Official website: https://rvsse.github.io/

The official workshop page lists this work as **Paper #14**.

## Disclaimer

This README summarizes the methodology and experimental findings reported in the associated paper. Dataset access, pretrained models, implementation details, and licensing are subject to the terms and availability of their respective sources.
