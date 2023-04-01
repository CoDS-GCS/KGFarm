<p align="center">
    <a href="https://www.mitacs.ca/en/projects/feature-discovery-system-data-science-across-enterprise">
      <img src="docs/graphics/KGFarm_logo.svg" width="550">
    </a>
</p>

### <p align="center"><b>A Holistic Platform for Data Preparation and Feature Discovery</b></p>
<p align="center">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue"/></a>
</p>

## 📐 System Design
<p align="center"><img src="docs/graphics/KGFarm.jpeg" alt="kgfarm" height="450" width="400"/></p>


<p align="justify">Data preparation and feature discovery are critical to improving model accuracy. However, data scientists often work independently and spend most of their time writing code for these steps without support for auto-learning from each other's work. KGFarm is the first holistic platform automating data preparation and feature discovery based on the semantics of seen pipelines applied to different datasets. This semantics is captured in a knowledge graph (KG).
KGFarm provides seamless integration with existing data science platforms to effectively enable scientific communities automatically discover and learn about each other's work. Thus, KGFarm enables data scientists to quickly automate pipelines with high accuracy for seen and unseen datasets. Our comprehensive evaluation used a KG constructed from Kaggle datasets and pipelines. KGFarm scales better than existing methods in recommending data cleaning, transformation, and feature selection while achieving better or comparable accuracy. During the demo, the audience will experience KGFarm with different datasets.</p>

## ⚡ Quick Start
Try the sample <b>[KGFarm Colab notebook](https://colab.research.google.com/drive/1u4z4EKGd8G1ju61Q3sPk5fH9BrMp8IRM?usp=sharing)</b> for a quick hands-on!

1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Connect to the [Stardog](https://www.stardog.com/) engine
```bash
stardog-admin server start
```
3. Run KGFarm's [<code>graph_builder</code>](feature_discovery/src/graph_builder/build.py):<br/>
generates [<code>farm.ttl</code>](https://github.com/CoDS-GCS/KGFarm/blob/645f12dfd63bae0bd319401c2cf10f8378dd6679/feature_discovery/src/graph_builder/farm.ttl) and uploads it to the [stardog server](https://cloud.stardog.com/)

```bash
cd feature_discovery/src/graph_builder
python build.py -db Database_name
```
4. Start using KGFarm APIs (checkout [<code>KGFarm_demo.ipynb</code>](KGFarm_demo.ipynb))

## 🚧 Roadmap

* [X] Automate Entity extraction
* [X] Predict Feature view with multiple Entities
* [X] Support Entity updation on the fly
* [X] Predict Features for Dataset enrichment
* [X] Recommend Data cleaning
* [X] Recommend Data transformation
* [X] Automate Feature selection

## 🧪 Experiments 

We compared KGFarm to several related system. More information regarding our evaluations per task is available below:
1. [Data transformation](experiments/results/evaluations%20KGFarm%20PVLDB%202023%20-%20Data%20transformation.pdf)
2. [Data cleaning](experiments/results/evaluations%20KGFarm%20PVLDB%202023%20-%20Data%20cleaning.pdf)
3. [Feature selection](experiments/results/evaluations%20KGFarm%20PVLDB%202023%20-%20Feature%20selection.pdf)

## <img src="docs/graphics/icons/youtube.svg" alt="youtube" height="20" width="29"> KGFarm Demo
<a href="https://rebrand.ly/kgfarm"><img src="docs/graphics/thumbnails/demo_thumbnail.jpeg"/></a>

## 🦾 Contributors
<p float="left">
 
  <img src="docs/graphics/CoDS.png" width="200"/> 

  <img src="docs/graphics/borealisAI.png" width="170"/>
</p>
