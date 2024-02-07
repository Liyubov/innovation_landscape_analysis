# Innovation landscape analysis

Repository contains code and sample of data for analysis of patents data. The methodological framework description contains description of embedding and dimensionality reduction methods.

# Method description 
Step 1. We first build the numerical representation of the
data. Depending on the structure of the data, such represen-
tation can be constructed using either: a) metadata in the categorical format (in the case of USPTO patents data the metadata consists of the
CPC categories of patents describing the relation towards
specific scientific area, which the patent is submitted), b) textual data of the abstracts. The BERT model for embedding was downloaded from https://github.com/google/patents-public-data/blob/master/models/BERT%20for%20Patents.md 

Step 2. Using the numerical high-dimensional data obtained at the first step we then build the low-dimensional representation through
applying the dimensionality reduction methods. Several methods are possible to apply at this stage include tSNE, diffusion maps methods.

Step 3.  Furthermore the clustering methods are applied to the high- and low-dimensional representation of data. The resulted data can as well further analyzed using methods from the mobility data analysis and general stochastic processes formalism adapted from https://arxiv.org/abs/2302.13054

## Load data
We analyze the data from open USPTO dataset.
