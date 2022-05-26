# KGFarm
<b>A Feature Discovery system for Machine learning workflows powered by Knowledge Graphs</b>
## 📐 System Design
![design](https://user-images.githubusercontent.com/40717058/162835808-3f99b48f-78f6-44c8-a431-88a09da43d7c.png)

## ⚡ Quick Start
- Installing dependencies
```bash
pip install -r requirements.txt
```
- Setup feature repository using feast
```bash
feast init feature_repo
```
- Connect to the [Stardog](https://www.stardog.com/) engine
```bash
stardog-admin server start
```
- Start using KGFarm APIs (checkout <code>KGFarm_notebook.ipynb</code>)

## 📋 Roadmap
* [X] Automate [Feature view](https://docs.feast.dev/getting-started/concepts/feature-view) generation
* [x] Predict [Entity](https://docs.feast.dev/v/v0.6-branch/user-guide/entities) consisting of a single join key
* [x] Predict Features for [Model training](https://docs.feast.dev/getting-started/quickstart#step-4-generating-training-data)
* [ ] Predict [Feature service](https://docs.feast.dev/getting-started/concepts/feature-service)
* [ ] Predict Composite keys for Joins
* [ ] Predict feature transformations
* [ ] Recommend Semantically similar features
* [ ] Recommend similar features by Content
* [ ] Detect Feature bias

## 🦾 Contributors
<p float="left">
 
  <img src="helpers/graphics/CoDS.png" width="200"/> 

  <img src="helpers/graphics/borealisAI.png" width="170"/>
</p>