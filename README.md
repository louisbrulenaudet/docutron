# Docutron Toolkit: detection and segmentation analysis for legal data extraction over documents
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue)

Docutron is a tool designed to facilitate the extraction of relevant information from legal documents, enabling professionals to create datasets for fine-tuning language models (LLM) for specific legal domains.

Legal professionals often deal with vast amounts of text data in various formats, including legal documents, contracts, regulations, and case law. Extracting structured information from these documents is a time-consuming and error-prone task. Docutron simplifies this process by using state-of-the-art computer vision and natural language processing techniques to automate the extraction of key information from legal documents.

![Docutron testing image](https://github.com/louisbrulenaudet/docutron/blob/main/preview.png?raw=true)

Whether you are delving into contract analysis, legal document summarization, or any other legal task that demands meticulous data extraction, Docutron stands ready to be your reliable technical companion, simplifying complex legal workflows and opening doors to new possibilities in legal research and analysis.

## Tech Stack
**Language:** Python +3.9.0

## Dependencies
The script relies on the following Python libraries:
-   Python
-   PyTorch
-   Detectron2
-   Cv2

## Installation
Clone the repo

```sh
git  clone  https://github.com/louisbrulenaudet/docutron.git
```

## Roadmap
- [x] Complete the first training and testing
- [x] Create the first dataset for labeling process
- [ ] Create a second version of the dataset in order to handle more cases
- [ ] Implementing in a structured architecture

## Citing this project

If you use this code in your research, please use the following BibTeX entry.
```BibTeX

@misc{louisbrulenaudet2023,
  author = {Louis Brulé Naudet},
  title = {Docutron Toolkit: detection and segmentation analysis for legal data extraction over documents},
  howpublished = {\url{https://github.com/louisbrulenaudet/docutron}},
  year = {2023}
}
```

## Feedback
If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).
