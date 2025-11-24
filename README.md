<h1 align="center">SuperQuadricOcc: Multi-Layer Gaussian Approximation of Superquadrics for Real-Time Self-Supervised Occupancy Estimation</h1>

<p align="center">
<strong>Seamie Hayes</strong><sup>1,2</sup>, 
<strong>Reenu Mohandas</strong><sup>1</sup>, 
<strong>Tim Brophy</strong><sup>1</sup>,
<strong>Alexandre Boulch</strong><sup>3</sup>,
<strong>Ganesh Sistu</strong><sup>1</sup>,
<strong>Ciaran Eising</strong><sup>1,2</sup>
</p>

<p align="center">
<sup>1</sup> D²iCE Research Centre, University of Limerick &nbsp;&nbsp; <sup>2</sup>Taighde Éireann – Research Ireland &nbsp;&nbsp; <sup>3</sup>Valeo.ai
</p>

<p align="center">
<a href="https://arxiv.org/abs/2511.17361">[arXiv]</a>
</p>

<img src="assets/superquadricocc2.jpg" width="100%">

Video visualizations are provided in the ```assets/``` folder

## Abstract
Semantic occupancy estimation enables comprehensive scene understanding for automated driving, providing dense spatial and semantic information essential for perception and planning. While Gaussian representations have been widely adopted in self-supervised occupancy estimation, the deployment of a large number of Gaussian primitives drastically increases memory requirements and is not suitable for real-time inference. In contrast, superquadrics permit reduced primitive count and lower memory requirements due to their diverse shape set. However, implementation into a self-supervised occupancy model is nontrivial due to the absence of a superquadric rasterizer to enable model supervision. Our proposed method, SuperQuadricOcc, employs a superquadric-based scene representation. By leveraging a multi-layer icosphere-tessellated Gaussian approximation of superquadrics, we enable Gaussian rasterization for supervision during training. On the Occ3D dataset, SuperQuadricOcc achieves a 75\% reduction in memory footprint, 124\% faster inference, and a 5.9\% improvement in mIoU compared to previous Gaussian-based methods, without the use of temporal labels. To our knowledge, this is the first occupancy model to enable real-time inference while maintaining competitive performance. The use of superquadrics reduces the number of primitives required for scene modeling by 84\% relative to Gaussian-based approaches. Finally, evaluation against prior methods is facilitated by our fast superquadric voxelization module.

## Code
Our code will be released soon

## Acknowledgement
This publication has emanated from research conducted with the financial support of Taighde Éireann – Research Ireland under Grant number 18/CRT/6049. For the purpose of Open Access, the author has applied a CC BY public copyright licence to any Author Accepted Manuscript version arising from this submission.
