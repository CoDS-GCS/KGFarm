# KGFarm
<b>A Feature Discovery system for Machine learning workflows powered by Knowledge Graphs</b>
## 📐 System Design
![design](https://user-images.githubusercontent.com/40717058/162835808-3f99b48f-78f6-44c8-a431-88a09da43d7c.png)

## ⚡ Quick Start
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Setup [<code>feature repository</code>]((https://github.com/CoDS-GCS/KGFarm/tree/main/feature_repo)) using [feast](https://feast.dev/)
```bash
feast init feature_repo
```
3. Connect to the [Stardog](https://www.stardog.com/) engine
```bash
stardog-admin server start
```
4. Run KGFarm's [<code>graph_builder</code>](https://github.com/CoDS-GCS/KGFarm/blob/main/feature_discovery/src/graph_builder/builder.py):<br/>
generates [<code>Farm.nq</code>](https://github.com/CoDS-GCS/KGFarm/blob/main/feature_discovery/src/graph_builder/Farm.nq) and uploads it to the [stardog server](https://cloud.stardog.com/)

```bash
cd feature_discovery/src/graph_builder
python builder.py
```
5. Start using KGFarm APIs (checkout [<code>KGFarm_notebook.ipynb</code>](https://github.com/CoDS-GCS/KGFarm/blob/main/KGFarm_notebook.ipynb))

## 🚧 Roadmap
- <b>List of deliverables for KGFarm version 0.1</b> [🔗](https://docs.google.com/presentation/d/14JigzSty4pwJaTXSNbo-SYZBcSaTqanlC4ETbGJVbTU/edit?usp=sharing)
* [x] Predict [Entities](https://docs.feast.dev/v/v0.6-branch/user-guide/entities) 
* [X] Predict [Feature views without Entities](https://docs.feast.dev/getting-started/concepts/feature-view#feature-views-without-entities)
* [X] Predict [Feature view](https://docs.feast.dev/getting-started/concepts/feature-view) with single [Entity](https://docs.feast.dev/v/v0.6-branch/user-guide/entities)
* [X] Predict [Feature view](https://docs.feast.dev/getting-started/concepts/feature-view) with multiple [Entities](https://docs.feast.dev/v/v0.6-branch/user-guide/entities)
* [x] Predict Features for [Model training](https://docs.feast.dev/getting-started/quickstart#step-4-generating-training-data)
* [ ] Support for [Entity aliasing](https://docs.feast.dev/getting-started/concepts/feature-view#entity-aliasing)
* [ ] Predict feature transformations
* [ ] Recommend semantically similar features
* [ ] Recommend similar features by content
* [ ] Detect Feature bias

## 📗 Useful resources
- [Feature discovery slides](https://docs.google.com/presentation/d/14JigzSty4pwJaTXSNbo-SYZBcSaTqanlC4ETbGJVbTU/edit?usp=sharing)
- [Research objectives](https://docs.google.com/document/d/1M_iWqk0YUscxXPl3UKJ0m83NAXdVOhVbUXnbKry4dSQ/edit?usp=sharing)
- [Feature discovery mitacs proposal](https://docs.google.com/document/d/1fWrp-IS9ZkKcOavcGDTr3cYx05xQag-H-PuFApZn1AY/edit?usp=sharing)

## 🦾 Contributors
<p float="left">
 
  <img src="helpers/graphics/CoDS.png" width="200"/> 

  <img src="helpers/graphics/borealisAI.png" width="170"/>
</p>