# KGFarm
<b>A Feature Discovery system for Machine learning workflows powered by Knowledge Graphs</b>
## 📐 System Design
![design](https://user-images.githubusercontent.com/40717058/162835808-3f99b48f-78f6-44c8-a431-88a09da43d7c.png)

## ⚡ Quick Start
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Setup feature repository using feast
```bash
feast init feature_repo
```
3. Connect to the [Stardog](https://www.stardog.com/) engine
```bash
stardog-admin server start
```
4. Start using KGFarm APIs (checkout <code>KGFarm_notebook.ipynb</code>)

## 🚧 Roadmap
- <b>List of deliverables for KGFarm version 0.1</b> [🔗](https://docs.google.com/document/d/1M_iWqk0YUscxXPl3UKJ0m83NAXdVOhVbUXnbKry4dSQ/edit#heading=h.flxu6q5t6n5w)
* [X] Automate generation of [Feature views without Entities](https://docs.feast.dev/getting-started/concepts/feature-view#feature-views-without-entities)
* [X] Automate [Feature view](https://docs.feast.dev/getting-started/concepts/feature-view) generation for single [Entity](https://docs.feast.dev/v/v0.6-branch/user-guide/entities)
* [ ] Automate [Feature view](https://docs.feast.dev/getting-started/concepts/feature-view) generation for multiple [Entities](https://docs.feast.dev/v/v0.6-branch/user-guide/entities)
* [x] Predict [Entities](https://docs.feast.dev/v/v0.6-branch/user-guide/entities) 
* [x] Predict Features for [Model training](https://docs.feast.dev/getting-started/quickstart#step-4-generating-training-data)
* [ ] Predict [Feature service](https://docs.feast.dev/getting-started/concepts/feature-service)
* [ ] Predict feature transformations
* [ ] Recommend semantically similar features
* [ ] Recommend similar features by content
* [ ] Detect Feature bias

## 🦾 Contributors
<p float="left">
 
  <img src="helpers/graphics/CoDS.png" width="200"/> 

  <img src="helpers/graphics/borealisAI.png" width="170"/>
</p>