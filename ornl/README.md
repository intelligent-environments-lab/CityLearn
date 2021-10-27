# ORNL
This package enables addition of buildings from the [Model America – data and models of every U.S. building](https://doi.ccs.ornl.gov/ui/doi/339) dataset to the CityLearn environment.

# Usage
1. Download building model IDF files from the [dataset repository](https://doi.ccs.ornl.gov/ui/doi/339).
2. Create relevant county directory in [data/idf/counties](data/idf/counties) to place downloaded IDF files. An example will be to create `data/idf/counties/TX_Austin` directory to place Austin, Texas related IDF files.
3. From the collection of IDFs in [data/idf/counties](data/idf/counties), specify the selected IDFs to be included in the CityLearn environment by updating [selected.json](data/idf/selected.json).
