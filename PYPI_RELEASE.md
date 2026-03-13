# PyPI Release Mini Guide (`softcpsrecsimulator`)

Quick checklist to publish a new version from this repo.

## 1) Pre-flight

1. Ensure your changes are committed and pushed to `master`.
2. Confirm GitHub secret exists:
   - `PYPI_API_TOKEN` in `Settings -> Secrets and variables -> Actions`.
3. Bump package version in:
   - `citylearn/__init__.py` (e.g., `0.1.1`).

## 2) Local sanity check (recommended)

```bash
python -m pip install --upgrade pip build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
```

## 3) Commit + push version bump

```bash
git add citylearn/__init__.py
git commit -m "release softcpsrecsimulator X.Y.Z"
git push origin master
```

## 4) Tag + release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Then in GitHub:
1. Open `Releases`
2. Create/Publish release from tag `vX.Y.Z`

This triggers workflow:
- `.github/workflows/pypi_deploy.yml`

## 5) Verify publication

1. Check Actions workflow success (`build` + `publish` jobs).
2. Check package page:
   - `https://pypi.org/project/softcpsrecsimulator/`
3. Test install:

```bash
pip install softcpsrecsimulator==X.Y.Z
python -c "from citylearn.citylearn import CityLearnEnv; print('ok')"
```

## 6) If token was exposed

1. Revoke token in PyPI immediately.
2. Create a new token.
3. Update GitHub `PYPI_API_TOKEN` secret.
