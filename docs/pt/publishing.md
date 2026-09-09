# Guia de Publicacao

O CityLearn e publicado no PyPI como `citylearn` e importado como `citylearn`.

Versao inglesa: [../publishing.md](../publishing.md).

## Responsavel por Releases

Responsavel default por releases: [@calofonseca](https://github.com/calofonseca).

## Bump de Versao

1. Atualizar `citylearn/__init__.py`.
2. Atualizar [releases.md](releases.md).
3. Atualizar paginas de referencia afetadas quando mudarem schema, acoes, observacoes, KPIs ou datasets.
4. Correr a validacao em [developer_guide.md](developer_guide.md).
5. Fazer commit e tag.

Exemplo:

```console
git add citylearn/__init__.py README.md docs
git commit -m "Release v0.4.3"
git tag -a v0.4.3 -m "Release v0.4.3"
git push --follow-tags origin master
```

## Workflow PyPI

1. Usar o projeto PyPI existente chamado `citylearn`.
2. Confirmar que o environment `pypi` do GitHub tem secrets `PYPI_USERNAME` e `PYPI_PASSWORD` validos.
3. Fazer push do commit e da tag.
4. Criar uma GitHub Release ou correr manualmente o workflow `Publish Python Package`.
5. O workflow `.github/workflows/pypi_deploy.yml` cria `dist/*` e envia para o PyPI.

## Check Local de Build

```console
.venv/bin/python -m pip install --upgrade pip build twine
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

## Disciplina de Release

| Tipo de versao | Quando usar |
|---|---|
| Patch | Features aditivas compativeis, fixes, docs e testes. |
| Minor | Nova capacidade grande do simulador ou mudanca de schema/API. |
| Major | Breaking changes amplos. |

Antes de publicar, garantir que `docs/releases.md` tem:

| Campo | Objetivo |
|---|---|
| Summary | O que mudou e porquê. |
| Release owner | Tag GitHub da pessoa responsavel pela release. |
| Dataset/schema impact | Risco de migracao para datasets e configs existentes. |
| Compatibility | Se algoritmos/wrappers precisam de mudancas. |
| Validation | Comandos ou simulacoes que passaram. |
| Migration notes | Acoes necessarias depois do upgrade. |
