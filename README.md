<div align="center">
	<h1>TokenDial: Continuous Attribute Control in Text-to-Video via Spatiotemporal Token Offsets</h1>
	<a href="https://arxiv.org/abs/2603.27520"><img src="https://img.shields.io/badge/arXiv-2512.10950-b31b1b" alt="arXiv"></a>
	<a href="https://tokendial.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
	<!-- <a href="#"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> -->
</div>

>
> Zhixuan Liu, Peter Schaldenbrand, Yijun Li, Long Mai, Aniruddha Mahapatra, Cusuh Ham, Jean Oh, Jui-Hsien Wang
> 
> Adobe Research, Carnegie Mellon University

<p align="center">
  <img src="assets/teaser.gif" alt="teaser">
</p>

### TL;DR: We turn pretrained text-to-video models into continuous video editors, enabling slider-style control over appearance and motion magnitude.
We present TokenDial, a framework for continuous, slider-style attribute control in pretrained text-to-video generation models. While modern generators produce strong holistic videos, they offer limited control over how much an attribute changes (e.g., effect intensity or motion magnitude) without drifting identity, background, or temporal coherence. TokenDial is built on the observation: additive offsets in the intermediate spatiotemporal visual patch-token space form a semantic control direction, where adjusting the offset magnitude yields coherent, predictable edits for both appearance and motion dynamics. We learn attribute-specific token offsets without retraining the backbone, using pretrained understanding signals: semantic direction matching for appearance and motion-magnitude scaling for motion.


## TODOs

- [ ] Release train/inference code
- [ ] Release pre-trained model weights
- [ ] Release Huggingface demo



## Citation

If you find this project helpful, please consider citing our work:

```bibtex
@misc{liu2026tokendialcontinuousattributecontrol,
  title={TokenDial: Continuous Attribute Control in Text-to-Video via Spatiotemporal Token Offsets}, 
  author={Zhixuan Liu and Peter Schaldenbrand and Yijun Li and Long Mai and Aniruddha Mahapatra and Cusuh Ham and Jean Oh and Jui-Hsien Wang},
  year={2026},
  eprint={2603.27520},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.27520}, 
}
```