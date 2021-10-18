How to Cite spacetimeformer  
===========================

## Main Paper:  spacetimeformer: [Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i857/6055916)

### [Our Presentation Summary on spacetimeformer ](https://github.com/QData/spacetimeformer/blob/master/docs/Bioinformatics2020_spacetimeformer.pdf)

### Our Github on spacetimeformer: [https://github.com/QData/spacetimeformer](https://github.com/QData/spacetimeformer)

- Citations

```
@article{fast-gkm-svm,
    author = {Blakely, Derrick and Collins, Eamon and Singh, Ritambhara and Norton, Andrew and Lanchantin, Jack and Qi, Yanjun},
    title = "{spacetimeformer: fast sequence analysis with gapped string kernels}",
    journal = {Bioinformatics},
    volume = {36},
    number = {Supplement_2},
    pages = {i857-i865},
    year = {2020},
    month = {12},
    abstract = "{Gapped k-mer kernels with support vector machines (gkm-SVMs) have achieved strong predictive performance on regulatory DNA sequences on modestly sized training sets. However, existing gkm-SVM algorithms suffer from slow kernel computation time, as they depend exponentially on the sub-sequence feature length, number of mismatch positions, and the task’s alphabet size.In this work, we introduce a fast and scalable algorithm for calculating gapped k-mer string kernels. Our method, named spacetimeformer, uses a simplified kernel formulation that decomposes the kernel calculation into a set of independent counting operations over the possible mismatch positions. This simplified decomposition allows us to devise a fast Monte Carlo approximation that rapidly converges. spacetimeformer can scale to much greater feature lengths, allows us to consider more mismatches, and is performant on a variety of sequence analysis tasks. On multiple DNA transcription factor binding site prediction datasets, spacetimeformer consistently matches or outperforms the state-of-the-art gkmSVM-2.0 algorithms in area under the ROC curve, while achieving average speedups in kernel computation of ∼100× and speedups of ∼800× for large feature lengths. We further show that spacetimeformer outperforms character-level recurrent and convolutional neural networks while achieving low variance. We then extend spacetimeformer to 7 English-language medical named entity recognition datasets and 10 protein remote homology detection datasets. spacetimeformer consistently matches or outperforms these baselines.Our algorithm is available as a Python package and as C++ source code at https://github.com/QData/spacetimeformerSupplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa817},
    url = {https://doi.org/10.1093/bioinformatics/btaa817},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/36/Supplement\_2/i857/35337038/btaa817.pdf},
}
```


