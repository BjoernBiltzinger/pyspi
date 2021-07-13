---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Installation
3ML brings together multiple instrument and fitting software packages into a common framework. Thus, installing all the pieces can be a bit of a task for the user. In order to make this a less painless process, we have packaged most of the external dependencies into conda (see below). However, if you want more control over your install, 3ML is available on PyPI via pip. If you have issues with the installs, first check that you have properly installed all the external dependencies that *you* plan on using. Are their libraries accessible on you system's standard paths? If you think that you have everything setup properly and the install does not work for you, please [submit an issue](https://github.com/threeML/threeML/issues) and we will do our best to find a solution.
