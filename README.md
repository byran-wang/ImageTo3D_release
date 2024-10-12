<p align="center">
  <b>ROAS6000J Project: Image/Text to 3D generation.</b>
  <br/>
  <img alt="preface" src="./assets/preface.gif" width="80%"/>
</p>

## Contact
- TA: swang457@connect.hkust-gz.edu.cn


## Codebase includes:
- The basic pipleline of Image/Text to 3D generation for objects without core implementation of diffusion model.
- Evaluation dataset.
- Dataset reading code.
- Evaluation code for 3D generation model.

## Basic Requirement
- Complete **core implementation** of diffusion model.
- Compare the performance of different methods on the same evaluation dataset, which should include:
  - **At least three open-source methods**, such as [Zero123](https://github.com/cvlab-columbia/zero123), [Wonder3D](https://github.com/xxlong0/Wonder3D), [CraftsMan](https://github.com/wyysf-98/CraftsMan), and others.
  - **At least two closed-source methods**, such as [CLAY](https://github.com/CLAY-3D/OpenCLAY), [Tripo3D](https://www.tripo3d.ai/), [One2345plus](https://sudo-ai-3d.github.io/One2345plus_page/) , and others available on Hugging Face or their official websites.
- For open-source methods, **adjust various hyperparameters**—such as text prompts, classification guidance scales, elevation angles, and physical properties—to compare generation results and summarize evaluation outcomes.

## Advanced Requirement
- **Different 3D Representations**: We encourage you to investigate and implement various advanced 3D representations beyond the conventional Neural Radiance Fields (NeRF). This exploration could include experimenting with cutting-edge techniques like Triplane representation from EG3D, Gaussian volumetric models as detailed in Gaussian Splatting, or 3D diffusion model which can be found at CraftsMan. An interesting example is CraftsMan, which employs a 3D native diffusion model based on your inputs, potentially offering coarse geometries in seconds  and more nuanced control over normal-based geometry refiner.
- **3D Generation Enhancement**: Despite significant advancements, existing methods still struggle with distorted shapes and noisy surfaces. Explore incorporating more information or prior knowledge to enhance the coarse results from the first stage of generation. For example, include more observing views from a video sequence to improve side or back generation results, or incorporate advanced foundation models, such as image-to-normal models, to refine geometry.
- **Train or Fine-Tune a Personalized 3D Generation Model**: From evaluating different methods, you may notice that state-of-the-art (SOTA) methods do not satisfy your requirements for specific categories, such as a particular dog, car, or a personalized subject. Explore training or fine-tuning a personalized 3D generation model.
- ** Free your mind**: Reward will be given for every unique exploration.

## News
- 12/Oct/2024: First version.


## Installation


The following steps have been tested on Ubuntu20.04.

- You must have an NVIDIA graphics card with at least 24GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.9`.
- (Optional, Recommended) Create a virtual environment:

```sh
python3 -m virtualenv venv
. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```

- Install `PyTorch`. We have tested on `torch2.0.1+cu118`, but other versions should also work fine.

```sh
# or torch2.0.1+cu118
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

- (Optional) `tiny-cuda-nn` installation might require downgrading pip to 23.0.1


## Evaluation Dataset

All the methods should be evaluated on the supplied dataset which locates in `./dataset`
<p align="center">
    <img src="./assets/dataset.png" alt="Image" width="80%"/>
</p>




