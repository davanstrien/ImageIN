

# Notebook/pipeline overview

Currently a lot of our code is inside notebooks. Ideally this code would be moved ot a pipelining tool like ZenML to make the pipeline more portable, reproducible and testable. In the meantime, this page documents the rough steps taken in our current pipeline. 

## Data 

For this particular project we start with data from the internet archive. The overflow steps in the date pipeline:

1. Identify items (books) from the Internet Archive
2. Extract references to individual pages from these items 
3. Sample this dataset and label the images manually 
4. Load a subset of the full dataset created in step 2 
5. Apply labels to the dataset in 4 using weak supervision via the Snorkel Python library

### Initial data gathering 

The internet archive has a [command line tool](https://archive.org/developers/internetarchive/cli.html) that can be used to interact with their collections. In this case we use the `search` command to identify books (texts) between 1800 and 1950 with words in the title that are likely to indicate the presence of illustrations. We store the outputs of this search in a JSON Lines file. 

``` bash
ia search "title:(illustrated OR illustrations OR picture OR pictures) AND mediatype:(texts) AND date:[1800-01-01 TO 1950-01-01]" -> itemlist.jsonl
```

This search results in `44,561` items. Since each book is made up of pages, we create a dataset that has a row for each page since this is the level at which our model will work. Splitting the data to the page level results in [https://huggingface.co/datasets/ImageIN/IA_unlabelled](https://huggingface.co/datasets/ImageIN/IA_unlabelled). This contains `1,0844,387` pages. 

### Labelling using weak supervision

The [IA_weak_pipeline.ipynb](https://github.com/davanstrien/ImageIN/blob/data-documentatioin/notebooks/IA_weak_pipeline.ipynb) notebook uses a subset of the unlabelled data and defines labelling functions to label the data using the [snorkel](https://snorkel.readthedocs.io/en/v0.9.7/) library. These labelling functions are Python functions that aim to predict whether a particular page image is illustrated or not based on some heuristics or rules. 

The power of snorkel is that it allows us to combine different labelling functions all of which may only label a subset of the data (and may make some mistakes) and generate a combined probability of the label for each data point. 




