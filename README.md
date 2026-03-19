<h1 align="center">SuperQuadricOcc: Real-Time Self-Supervised Semantic Occupancy Estimation with Superquadric Volume Rendering</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=3fffnjYAAAAJ&hl=en"><strong>Seamie Hayes</strong></a><sup>1,2</sup>,
  <a href="https://scholar.google.com/citations?user=iJ3qFGAAAAAJ&hl=en"><strong>Alexandre Boulch</strong></a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=HTfERCsAAAAJ&hl=en"><strong>Andrei Bursuc</strong></a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=Ku6jvh0AAAAJ&hl=en"><strong>Reenu Mohandas</strong></a><sup>1</sup>, <br>
  <a href="https://scholar.google.com/citations?user=dDgm87sAAAAJ&hl=en"><strong>Tim Brophy</strong></a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=356ahmwAAAAJ&hl=en"><strong>Ganesh Sistu</strong></a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=aH6w8VcAAAAJ&hl=en"><strong>Ciaran Eising</strong></a><sup>1,2</sup>
</p>


<p align="center">
<sup>1</sup> D²iCE Research Centre, University of Limerick &nbsp;&nbsp; <sup>2</sup> Taighde Éireann – Research Ireland &nbsp;&nbsp; <sup>3</sup> Valeo.ai
</p>

<p align="center">
<a href="https://arxiv.org/abs/2511.17361">
  <img src="https://img.shields.io/badge/arXiv-2511.17361-b31b1b.svg">
</a>
</p>

<img src="assets/superquadricocc.jpg" width="100%">

Video visualizations are provided in the ```assets/``` folder

## Abstract
_Self-supervision for semantic occupancy estimation is appealing as it removes the labour-intensive manual annotation, thus allowing one to scale to larger autonomous driving datasets. Superquadrics offer an expressive shape family very suitable for this task, yet their deployment in a self-supervised setting has been hindered by the lack of efficient rendering methods to bridge the 3D scene representation and 2D training pseudo-labels. To address this, we introduce SuperQuadricOcc, the first self-supervised occupancy model to leverage superquadrics for scene representation. To overcome the rendering limitation, we propose a real-time volume renderer that preserves the fidelity of the superquadric shape during rendering. It relies on spatial superquadric–voxel indexing, restricting each ray sample to query only nearby superquadrics, thereby greatly reducing memory usage and computational cost. Using drastically fewer primitives than previous Gaussian-based methods, SuperQuadricOcc achieves state-of-the-art performance on the Occ3D-nuScenes dataset, while running at real-time inference speeds with substantially reduced memory footprint._

## Code
Our code will be released soon

## Acknowledgement
This publication has emanated from research conducted with the financial support of Taighde Éireann – Research Ireland under Grant number 18/CRT/6049. For the purpose of Open Access, the author has applied a CC BY public copyright licence to any Author Accepted Manuscript version arising from this submission.

I would like to thank the authors of the following open-source projects:<br>
[GaussianFlowOcc](https://github.com/boschresearch/GaussianFlowOcc)<br>
[PartGS](https://github.com/zhirui-gao/PartGS)<br>
[QuadricFormer](https://github.com/zuosc19/QuadricFormer)<br>
[PyTorch3D](https://github.com/facebookresearch/pytorch3d)<br>

## Citation
```
@misc{hayes2026superquadricoccrealtimeselfsupervisedsemantic,
      title={SuperQuadricOcc: Real-Time Self-Supervised Semantic Occupancy Estimation with Superquadric Volume Rendering}, 
      author={Seamie Hayes and Alexandre Boulch and Andrei Bursuc and Reenu Mohandas and Ganesh Sistu and Tim Brophy and Ciaran Eising},
      year={2026},
      eprint={2511.17361},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.17361}, 
}
```
