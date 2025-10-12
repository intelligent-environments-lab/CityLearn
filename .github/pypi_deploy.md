# How to release a new version to PyPi

1. Update the version number in [`citylearn.__init__`](../citylearn/__init__.py). The version format is `X.Y.Z` where `X` is bumped up for major releases that will come with breaking changes for previous versions, `Y` is bumped up for new features that do not cause breaking changes and `Z` is bumped up for bug fixes.

2. Commit and push the updated `citylearn.__init__` file:
    ```shell
    git add citylearn/__init__.py
    git commit -m "Updated version to vX.Y.Z"
    git push
    ```

3. Create a new tag for the updated version and push to the remote:
    ```shell
    git tag vX.Y.Z -m "<A message describing the changes introduced in the new tag>"
    git push upstream --tags
    ```

4. Go to the [tags page on GitHub](https://github.com/intelligent-environments-lab/CityLearn/tags) and create a new release for the new `vX.Y.Z` tag. Once the release has been created, the [`workflows/pypi_deply.yml`](workflows/pypi_deploy.yml) `GitHub Actions` will be automatically triggered to upload a new release to the [CityLearn PyPi project](https://pypi.org/project/CityLearn/).