# A Model-Agnostic Approach to Quantifying the Informativeness of Explanation Methods for Time Series Classification

Code repository for workshop paper in [AALTD](https://project.inria.fr/aaltd20/accepted-papers/), [ECML/PKDD 2020](https://ecmlpkdd2020.net/)

## Abstract

In this paper we focus on explanation methods for time series classification. In particular, we aim to quantitatively assess and rank different explanation methods based on their informativeness. In many applications, it is important to understand which parts of the time series are informative for the classification decision. For example, while doing a physio exercise, the patient receives feedback on whether the execution is correct or not (classification), and if not, which parts of the motion are incorrect (explanation), so they can take remedial action. Comparing explanations is a non-trivial task. It is often unclear if the output presented by a given explanation method is at all informative (i.e., relevant for the classification task) and it is also unclear how to compare explanation methods side-by-side. While explaining classifiers for image data has received quite some attention, explanation methods for time series classification are less explored. We propose a model-agnostic approach for quantifying and comparing different saliency-based explanations for time series classification. We extract importance weights for each point in the time series based on learned classifier weights and use these weights to perturb specific parts of the time series and measure the impact on classification accuracy. By this perturbation, we show that explanations that actually highlight discriminative parts of the time series lead to significant changes in classification accuracy. This allows us to objectively quantify and rank different explanations. We provide a quantitative and qualitative analysis for a few well known UCR datasets.

## Full Paper

[ResearchGate Link](https://www.researchgate.net/publication/347873962_A_Model-Agnostic_Approach_to_Quantifying_the_Informativeness_of_Explanation_Methods_for_Time_Series_Classification)

## Citation
https://doi.org/10.1007/978-3-030-65742-0_6

```
@InProceedings{10.1007/978-3-030-65742-0_6,
author="Nguyen, Thu Trang
and Le Nguyen, Thach
and Ifrim, Georgiana",
editor="Lemaire, Vincent
and Malinowski, Simon
and Bagnall, Anthony
and Guyet, Thomas
and Tavenard, Romain
and Ifrim, Georgiana",
title="A Model-Agnostic Approach to Quantifying the Informativeness of Explanation Methods for Time Series Classification",
booktitle="Advanced Analytics and Learning on Temporal Data",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="77--94",
abstract="In this paper we focus on explanation methods for time series classification. In particular, we aim to quantitatively assess and rank different explanation methods based on their informativeness. In many applications, it is important to understand which parts of the time series are informative for the classification decision. For example, while doing a physio exercise, the patient receives feedback on whether the execution is correct or not (classification), and if not, which parts of the motion are incorrect (explanation), so they can take remedial action. Comparing explanations is a non-trivial task. It is often unclear if the output presented by a given explanation method is at all informative (i.e., relevant for the classification task) and it is also unclear how to compare explanation methods side-by-side. While explaining classifiers for image data has received quite some attention, explanation methods for time series classification are less explored. We propose a model-agnostic approach for quantifying and comparing different saliency-based explanations for time series classification. We extract importance weights for each point in the time series based on learned classifier weights and use these weights to perturb specific parts of the time series and measure the impact on classification accuracy. By this perturbation, we show that explanations that actually highlight discriminative parts of the time series lead to significant changes in classification accuracy. This allows us to objectively quantify and rank different explanations. We provide a quantitative and qualitative analysis for a few well known UCR datasets.",
isbn="978-3-030-65742-0"
}


```
