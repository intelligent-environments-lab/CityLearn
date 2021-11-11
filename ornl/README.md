# ORNL
This package enables addition of buildings from the [Model America – data and models of every U.S. building](https://doi.ccs.ornl.gov/ui/doi/339) dataset to the CityLearn environment.

# Usage
1. Download building model IDF files from the [dataset repository](https://doi.ccs.ornl.gov/ui/doi/339).
2. From the collection of downloaded IDFs, specify IDFs to be included in the CityLearn environment by updating [selected.json](data/idf/selected.json).
3. Modify the `idf_directory` path and execute the [run.sh](run.sh) shell script to update CityLearn with selected buildings.