python_sources(name="lib", sources=["**/*.py"])

python_distribution(
    name="refuge-packaged",
    dependencies=[
        ":lib"
    ],
    sdist=True,
    wheel=True,
    provides=python_artifact(
        name="refuge",
    ),
    generate_setup=False,
)

resources(
    name="resources",
    sources=[
        "LICENSE",
        "LICENSE.ADDITIONAL-TERMS",
        "README.md",
        "pyproject.toml",
        "refuge/py.typed",
    ],
)