# ImageIN

This repository contains work done as part of the 2022 edition of the [Full Stack Deep Learning course](https://fullstackdeeplearning.com/course/2022/). The team contributing to this work consisted of [@zacbouhnik](https://github.com/zacbouhnik), [@PawPod](https://github.com/PawPod), [@fsmitskamp](https://github.com/fsmitskamp) and [@davanstrien](https://github.com/davanstrien).


## Project aim

**tl;dr** this project develops a machine learning model which can tell you whether a page of a historical book contains an illustration or not. 

![A digitized image from a historic book. The page only contains text](https://iiif.archivelab.org/iiif/memoirslettersof01bernuoft$105/full/250,/0/default.jpg) ![A digitized image from a historic book. The page only contains an illustration](https://iiif.archivelab.org/iiif/memoirslettersof01bernuoft$10/full/250,/0/default.jpg)

### Background

Over the past couple of decades libraries, archives, museums, and other organisations have increasingly digitised content they hold. This includes a large number of digitised books. Some of these books contain illustrations, it would be nice to be able to identify pages which contain illustrations so that:

- you know what (if any) other machine learning models you might use on that page (Optical character Recognition, layout detection etc.)
- so you can extract datasets of visual content from books 
- as part of an image search pipeline
- so you can look at nice pictures more easily
- ... 


### Project data 

For this project we primarily used data from the Internet Archive.

| Dataset                                                                             | Description                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ImageIN/IA_unlabelled](https://huggingface.co/datasets/ImageIN/IA_unlabelled)      | This a dataset of pages from historic digitised books held by the Internet Archive. To increase the presence of illustrations in downstream datasets the words 'illustration' are used in the intial search |
| [ImageIn_annotations](https://huggingface.co/datasets/ImageIN/ImageIn_annotations) | This dataset is sampled from the full unlabelled dataset and contains hand annotated labels indicating if a page is 'illustrated' or 'not-illustrated'                                                                                                                                                                                                           |
|                                                                                     |                                                                                                                                                                                                             |
|                                                                                     |                                                                                                                                                                                                             |



### Approach and tools used 

The project has a number of enablers and constrains which informed the approach taken. 

#### Enablers

- Open access data: libraries, archives and museums make large amounts of their data available under permissive licences (primarily Creative Commons)
- IIIF (more on this below)
- Existing metadata: existing metadata for collections items can enable filtering and retrieval of relevant material more easily. 

#### Constaints

- Labelled data
- Computation for inference

#### Approach



#### Tools

- **Snorkel** is used for helping generate labels using labelling functions
- **Weights and Biases** is used for tracking model training
- **🤗 datasets** is used for processing our data and moving it between machines via the Hugging Face hub 
- **🤗 transformers:** we use transformer implementation of the computer vision models we use and the `Trainer` class for model training. 
- **IIIF**: see below 


#### IIIF (International Image Interoperability Framework) 

#### Outputs 

Our Hugging Face hub organisation ([https://huggingface.co/ImageIN](https://huggingface.co/ImageIN)) hold outputs generated from this project. 


### Further work and possible enhancements
