# Final Research Methodology

**Project:** Recognition-Aware Low-Light Image Enhancement for Improved Object Detection  
**Degree:** M.Tech Minor Project

---

## 1. Problem Formulation & Motivation

Low-light imaging is degraded by:
1. Low photon count leading to severe attenuation of scene luminance ($L < 30$).
2. Low signal-to-noise ratio (SNR) amplifying Poisson-Gaussian sensor noise.
3. Contrast compression obscuring object boundaries, contours, and semantic textures.

Standard object detectors (such as D-FINE, Faster R-CNN, DETR) are trained predominantly on well-illuminated datasets (e.g., MS-COCO, ImageNet). When evaluated on low-light inputs, high-frequency shadow noise activates false-positive priors, while darkened foreground targets fall below the confidence activation threshold.

**Core Research Question:**  
*Does illumination enhancement before object detection consistently improve downstream recognition, or can aggressive enhancement amplify noise and generate spurious detections?*

---

## 2. Research Architecture & Pipeline

```
 +---------------------------------------------------------------------------------------------+
 |                                  Input Low-Light Image                                      |
 +---------------------------------------------------------------------------------------------+
                                                 |
             +-----------------------------------+-----------------------------------+
             |                                   |                                   |
             v                                   v                                   v
   [Condition B: Zero-DCE]           [Condition C: IGR-Net]           [Condition D: Retinexformer]
   • 7-Layer DCE-Net                 • Proposed Novel Model           • Adapted Architecture
   • Non-Reference Curves            • Max-RGB Illumination Guided    • IG-MSA Self-Attention
   • Spatial/Exposure Losses         • Noise-Aware Spatial Attention  • Physical Recombination
   • 79,416 parameters               • 62,060 parameters              • 28,708 parameters
             |                                   |                                   |
             +-----------------------------------+-----------------------------------+
                                                 |
                                                 v
                     +-------------------------------------------------------+
                     |         Unified Downstream Object Detector            |
                     |         D-FINE-S (Official Pretrained MS-COCO)        |
                     |         • HGNetv2 Backbone + HybridEncoder            |
                     |         • Fine-Grained Distribution Refinement        |
                     |         • 10,321,877 parameters (strict checkpoint)   |
                     +-------------------------------------------------------+
                                                 |
                                                 v
                     +-------------------------------------------------------+
                     |          Comparative Recognition Analysis             |
                     |   Bounding Boxes | Confidences | Luminance | Contrast |
                     +-------------------------------------------------------+
```

---

## 3. Mathematical Foundations of IGR-Net (Proposed Novel Contribution)

### A. Retinex Decomposition Model
An observed low-light image $I(x,y) \in \mathbb{R}^{3 \times H \times W}$ is decomposed into:
$$I(x,y) = R(x,y) \odot L(x,y)$$
where $R(x,y)$ represents the illumination-invariant reflectance carrying structural boundaries, and $L(x,y)$ represents the non-uniform illumination map.

### B. Max-RGB Prior Initialization
Instead of iterative estimation, the initial illumination prior is initialized via the channel-wise maximum:
$$L_{\text{init}}(x,y) = \max_{c \in \{R,G,B\}} I_c(x,y)$$

### C. Local-Global Multi-Scale Module (LGM)
To capture both local edge gradients and broad contextual lighting, features $F \in \mathbb{R}^{C \times H \times W}$ are processed by:
$$F_{\text{local}} = \text{ReLU}(\text{BN}(\text{Conv}_{3\times3}(F)))$$
$$F_{\text{dilated}} = \text{ReLU}(\text{BN}(\text{Conv}_{3\times3, d=2}(F)))$$
$$F_{\text{fused}} = \text{Conv}_{1\times1}([F_{\text{local}}, F_{\text{dilated}}])$$
$$F_{\text{out}} = F_{\text{fused}} \odot \sigma(\text{MLP}(\text{GAP}(F))) + F$$

### D. Spatially-Varying Noise-Aware Attention (NAA)
In low-illumination regions, high-frequency variations typically represent sensor noise rather than texture. The noise-aware weight map is formulated as:
$$M_{\text{noise}} = \sigma(\text{Conv}_{3\times3}(\text{ReLU}(\text{Conv}_{3\times3}(F))))$$
$$W_{\text{suppress}} = 1.0 - 0.5 \cdot \left[(1.0 - L_{\text{norm}}) \odot M_{\text{noise}}\right]$$
$$F_{\text{denoised}} = F \odot W_{\text{suppress}}$$
This prevents shadow noise from exciting downstream detector priors.

---

## 4. Downstream Detection Pipeline: D-FINE-S

To benchmark recognition performance without architectural bias, all enhanced images are passed through the official **D-FINE-S** detector:
1. **Backbone:** Hierarchical Golden Neural Network v2 (HGNetv2 B0 stage) providing multi-scale pyramid representations.
2. **Hybrid Encoder:** Integrates intra-scale self-attention and cross-scale PAN-style feature fusion.
3. **Fine-Grained Distribution Refinement Decoder:** Employs probability distribution regression over box boundaries, achieving high localization precision.
4. **Unified Protocol:** Identical confidence threshold ($\tau = 0.25$) and NMS IoU threshold ($\text{IoU} = 0.45$) applied across all branches.
