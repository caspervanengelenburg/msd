# MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes

This repository contains the source code accompanying the paper:

[MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes](https://arxiv.org/abs/2407.10121) </br>
[**Casper van Engelenburg**](https://www.tudelft.nl/staff/c.c.j.vanengelenburg/?cHash=a72f1da92639fa8301893a08d4b49da1)<sup>*</sup>,
[**Fatemeh Mostafavi**](https://www.tudelft.nl/staff/f.mostafavi/?cHash=db57104f8776c2a1522b91c039845e84),
**Emanuel Kuhn**,
[**Yuntae Jeon**](https://yuntaej.github.io/),
**Michael Franzen**,
[**Matthias Standfest**](https://standfest.science/), 
[**Jan C. van Gemert**](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/computer-vision-lab/people/jan-van-gemert),
[**Seyran Khademi**](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/computer-vision-lab/people/seyran-khademi)

![msd_examples.png](figures/msd_examples.png)

In: Proceedings of the **IEEE/CVF European Conference on Computer Vision (ECCV), 2024**

### Abstract

*Diverse and realistic floor plan data are essential for the development of useful computer-aided methods in architectural design. 
Today's large-scale floor plan datasets predominantly feature simple floor plan layouts, typically representing single-apartment dwellings only.
To compensate for the mismatch between current datasets and the real world, we develop **Modified Swiss Dwellings** (MSD) - the first large-scale floor plan dataset that contains a significant share of layouts of multi-apartment dwellings. 
MSD features over 5.3K floor plans of medium- to large-scale building complexes, covering over 18.9K distinct apartments.
We validate that existing approaches for floor plan generation, while effective in simpler scenarios, cannot yet seamlessly address the challenges posed by MSD. 
Our benchmark calls for new research in floor plan machine understanding. 
Code and data are open.*

![msd_representation.png](figures/msd_representation.png)

## MSD datasets

The complete dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings). 
More details on the dataset can be found there.

**The graph as base data structure**</br>
The floor plans come in various linked modalities: image, geometry, and graph. The **main data container is the graph** (`networkx.Graph()` or `torch_geometric.data.Data()`) on-top-of which the room shapes and types (as node-level attributes), the connectivity types (as edge-level attributes) and the full image (as graph-level attribute) are modelled. 
While most floor plan datasets are not structured in this way, we believe that the graph captures most precisely the underlying spatial structure of the floor plan's layout. 
Whatever element you need on top of the graph, you can add (by populating the nodes, edges, or by adding a graph attribute) yourself. 
And if you don't need anything besides the topology of the floor plan, just use the bare graph: `networkx.Graph().nodes(data=False)` and `networkx.Graph().edges(data=False)`.

**Structure**
- [Data guidelines](NB - Usage Guidelines.ipynb): step-by-step notebook on how to use the data
- [Plotting functions](plot.py): some functions that make it easier to make nice plots
- [Constants](constants.py): contains lists and dictionaries of things like the color settings that we use
- [Utilities](utils.py): some useful utility functions


## Floor plan generation

The code for the models will be released soon!

- [ ] Include code for Graph-informed U-Net: will be in [this](unet) folder. Development branch [here](https://github.com/caspervanengelenburg/msd/tree/yt): 

```bash
git checkout yt
```

- [ ] Include code for Modified HouseDiffusion: will be in [this](mhd) folder. Development branch [here](https://github.com/caspervanengelenburg/msd/tree/wip-house-diffusion-msd): 

```bash
git checkout wip-house-diffusion-msd
```

## Cite

<pre><code>@misc{vanengelenburg2024msd,
      title={MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes},
      author={van Engelenburg, Casper and Mostafavi, Fatemeh and Kuhn,
              Emanuel and Jeon, Yuntae and Franzen, Michael and Standfest,
              Matthias and van Gemert, Jan and Khademi, Seyran},
      year={2024}
      eprint={2407.10121}
      archivePrefix={arXiv}
      primaryClass={cs.CV}
}</code></pre>
