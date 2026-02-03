# Deep Learning: Article study
## Explaining and Harnessing Adversarial Examples

This repository contains the code related to the review by Adonis Jamal and Jean-Vincent Martini of the article "Explaining and Harnessing Adversarial Examples", written Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy, for the Deep Learning course of MVA master's degree (ENS Paris-Saclay) and CentraleSupélec.

Specifically, this code implements the Fast Gradient Sign Method (FGSM) to generate adversarial examples in object detection using mostly the YOLO11n model on the VOC dataset.

The scripts provided in this repository are as follows:
- ```convert_voc_to_yolo_format.py```: a script to convert the VOC dataset into YOLO format.
- ```create_adv_dataset.py```: a script to create a dataset of adversarial examples using FGSM attack. The parameter $\varepsilon$ can be adjusted to control the perturbation level.
- ```eval_for_base_model.py```: a script to evaluate the performance of a YOLO11n model on a VOC-like dataset. The parameter $\varepsilon$ can be adjusted to evaluate the model on adversarial examples with specific perturbation levels. 
- ```eval_for_ft_model.py```: a script to evaluate the performance of a fine-tuned YOLO11n model on a VOC-like dataset. The parameter $\varepsilon$ is similar to the one in ```eval_for_base_model.py```. This script is basically a faster version of the previous one but only for fine-tuned YOLO models on VOC (using Ultralytics built-in functions).
- ```eval_for_ssd.py```: a script to evaluate the performance of a SSDlite MobileNetV3 model on a VOC-like dataset. The parameter $\varepsilon$ is similar to the one in ```eval_for_base_model.py```. 
- ```print_bbxoes.py```: a utility script to print the bounding boxes of the images of a dataset.
- ```training.py```: a script to fine-tune a YOLO11n model on a VOC-like dataset with Ultralytics.
- ```notebooks/```: a folder containing Jupyter notebooks used for experimentation and visualization used in the report.

```results/``` is a folder containing the results of the evaluations in .json format. ```results_colab/``` is a folder containing the metrics obtained when fine-tuning the models on Google Colab.

```report/``` is a folder containing the LaTeX source code of the report and ```slides/``` is a folder containing the LaTeX source code of the presentation slides.