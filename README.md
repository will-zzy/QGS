# Quadratic Gaussian Splatting for Efficient and Detailed Surface Reconstruction

[Project page](https://quadraticgs.github.io/QGS/) | [Paper](https://arxiv.org/pdf/2411.16392) | [Quadric Surfel Rasterizer (python)](./quadratic_demo.py)

<div style="text-align: center;">
  <figure style="margin: 0;">
    <img src="assets/teaser2.png" alt="Image 1" width="1000">
  </figure>
</div>

This repo contains the official implementation for the paper “Quadratic Gaussian Splatting for Efficient and Detailed Surface Reconstruction.”
Following [2DGS](https://github.com/hbb1/2d-gaussian-splatting), we also provide a [Python demo](./quadratic_demo.py) that demonstrates the differentiable rasterization process for quadratic surfaces:
<div style="display: flex; gap: 10px;">
  <figure style="margin: 0;">
    <img src="assets/QGS_demo_convex.gif" alt="Image 1" width="250">
  </figure>
  <figure style="margin: 0;">
    <img src="assets/QGS_demo_saddle1.gif" alt="Image 2" width="250">
  </figure>
  <figure style="margin: 0;">
    <img src="assets/QGS_demo_saddle2.gif" alt="Image 3" width="250">
  </figure>
</div>

## New Feature
- 2025/08/11: We replaced the original rectangular bounding box with a more compact truncated cone-shaped bounding box, which significantly reduces the invalid rendering area and achieves a two-fold speedup.

<div align="center">
  <img src="assets/bbox.png" width="600" />
</div>

<div align="center">

|            | Mip-NeRF 360  |       |TNT            |       |
|:-----------|:-------------:|:-----:|:-------------:|:-----:|
|            | Training time | FPS   | Training time | FPS   |
| 2DGS       | 1h5min        | 15.34 | 34min         | 31.66 |
| QGS        | 1h48min       | 7.61  | 2h            | 13.27 |
| QGS w/ TB  | 1h13min       | 14.15 | 43min         | 25.36 |
</div>



## Installation

```dockerfile
# download
git clone https://github.com/will-zzy/QGS.git

conda env create -f environment.yml
```

## Training

To train a scene, simply use

```
python train.py --conf_path ./config/base.yaml # or DTU.yaml/TNT.yaml
```

In `base.yaml`, you can adjust all configurable parameters, with most parameters remaining consistent with 2DGS. Furthermore, we have briefly experimented with curvature-related losses, such as curvature distortion loss and curvature flatten loss. Unfortunately, their performance was not satisfactory.

You need to modify the `root_dir` in the `xxx.yaml file to point to your dataset directory, for example:
```
xxx.yaml
└── root_dir
    ├── images
    └── sparse/0
```

**Tips for adjusting the parameters on your own dataset:**

- We observed that setting `pipeline.depth_ratio=1` enhances rendering quality. Additionally, by employing per-pixel reordering, we effectively eliminate the "disk-aliasing" artifacts present in 2DGS when using `depth_ratio=1`. Therefore, we recommend using `pipeline.depth_ratio=1` when aiming to improve rendering quality. 
- In most scenarios, we recommend adjusting the `optimizer.densify_grad_threshold` and `optimizer.lambda_dist` parameters to achieve better reconstruction. The former controls the number of Gaussian primitives, while the latter controls the compactness of the primitives.
- For large scenes, especially aerial or street views, we suggest adjusting the number of training iterations based on the number of images. We provide `TNT_Courthouse.yaml` as an example.

## Testing

To extract scene geometry, simply use:

```dockerfile
python render.py --conf_path ./config/base.yaml
```

In the `pipeline` section of the `base.yaml` configuration file, you can set various parameters for mesh extraction, maintaining the same meanings as those in 2DGS.

## Evaluation

For geometry reconstruction on the DTU dataset, please download the preprocessed data from [Drive](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) or [Hugging Face](https://huggingface.co/datasets/dylanebert/2DGS). You also need to download the ground truth [DTU point cloud](https://roboimagedata.compute.dtu.dk/?page_id=36).

Next, modify the `DTU.yaml` configuration file by setting the `load_model_path` to the path of your trained model and `dataset_GT_path` to the path of the ground truth dataset. After making these changes, simply execute the following commands to perform the evaluation:

```docker
python scripts/eval_dtu/eval.py --conf_path ./config/DTU.yaml
```

For the TNT dataset, please download the preprocessed [TNT_data](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main). Additionally, you need to download the ground truth [TNT_GT](https://www.tanksandtemples.org/download/).

Next, modify the `TNT.yaml` configuration file by setting `load_model_path` to the path of your trained model and `dataset_GT_path` to the path of the ground truth dataset. After making these changes, simply execute the following commands to perform the evaluation:

```
python scripts/eval_tnt/run.py --conf_path ./config/TNT.yaml -m <path to pre-trained model>
```

We also provide [DTU Evaluation Results](https://drive.google.com/file/d/1QEAQli7uI_t0m1IpkcZ54pRO_OUyNJIl/view?usp=sharing)
<details>
<summary><span style="font-weight: bold;">Table Results</span></summary>

Chamfer distance on DTU dataset (lower is better)

|           | 24   | 37   | 40   | 55   | 63   | 65   | 69   | 83   | 97   | 105  | 106  | 110  | 114  | 118  | 122  | Mean |
|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| Paper     | 0.38 | 0.62 | 0.37 | 0.38 | 0.75 | 0.55 | 0.51 | 1.12 | 0.68 | 0.61 | 0.46 | 0.58 | 0.35 | 0.41 | 0.40 | 0.545 |
| Reproduce | 0.38 | 0.64 | 0.35 | 0.34 | 0.77 | 0.55 | 0.52 | 1.11 | 0.68 | 0.60 | 0.43 | 0.58 | 0.35 | 0.41 | 0.37 | 0.539 |
</details>


and [TNT Evaluation Results](https://drive.google.com/file/d/19fm64lJvzpOgStppS-po8Ohpr5FC5How/view?usp=sharing)
<details>
<summary><span style="font-weight: bold;">Table Results</span></summary>

F1 scores on TnT dataset (higher is better)

|           | Barn  | Caterpillar | Ignatius | Truck  | Meetingroom | Courthouse | Mean | 
|-----------|-------|-------------|----------|--------|-------------|------------|------|
| paper     | 0.55  | 0.40        | 0.81     | 0.64   | 0.31        | 0.28       | 0.50 |
| Reproduce | 0.56  | 0.41        | 0.80     | 0.67   | 0.39        | 0.27       | 0.52 |
</details>

## Acknowledgements

This project is built upon [2DGS](https://github.com/hbb1/2d-gaussian-splatting). The TSDF fusion for extracting mesh is based on [Open3D](https://github.com/isl-org/Open3D). The rendering script for MipNeRF360 is adopted from [Multinerf](https://github.com/google-research/multinerf/), while the evaluation scripts for DTU and Tanks and Temples dataset are taken from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation), respectively. We thank all the authors for their great repos.

## Citation

If you find our code or paper helps, please consider citing:
```bibtex
@misc{zhang2024quadraticgaussiansplattingefficient,
      title={Quadratic Gaussian Splatting for Efficient and Detailed Surface Reconstruction}, 
      author={Ziyu Zhang and Binbin Huang and Hanqing Jiang and Liyang Zhou and Xiaojun Xiang and Shunhan Shen},
      year={2024},
      eprint={2411.16392},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.16392}, 
}
```





