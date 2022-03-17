---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Contributing 

Contributions to ```PySPI``` are always welcome. They can come in the form of:

## Issues

Please use the [Github issue tracking system for any bugs](https://github.com/BjoernBiltzinger/pyspi/issues), for questions, bug reports and or feature requests.

## Add to Source Code

To directly contribute to the source code of ```PySPI```, please fork the Github repository, add the changes to one of the branches in your forked repository and then create a [pull request to the master of the main repository](https://github.com/BjoernBiltzinger/pyspi/pulls) from this branch. Code contribution is welcome for different topics:

### Add Functionality

If ```PySPI``` is missing some functionality that you need, you can either create an issue in the Github repository or add it to the code and create a pull request. Always make sure that the old tests do not break and adjust them if needed. Also please add tests and documentation for the new functionality in the pyspi/test folder. This ensures that the functionality will not get broken by future changes to the code and other people will know that this feature exists.

### Code Improvement

You can also contribute code improvements, like making calculations faster or improve the style of the code. Please make sure that the results of the software do not change in this case.

### Bug Fixes

Fixing bugs that you found or that are mentioned in one of the issues is also a good way to contribute to ```PySPI```. Please also make sure to add tests for your changes to check that the bug is gone and that the bug will not recur in future versions of the code.

### Documentation

Additions or examples, tutorials, or better explanations are always welcome. To ensure that the documentation builds with the current version of the software, we are using [jupytext](https://jupytext.readthedocs.io/en/latest/) to write the documentation in Markdown. These are automatically converted to and executed as jupyter notebooks when changes are pushed to Github. 


