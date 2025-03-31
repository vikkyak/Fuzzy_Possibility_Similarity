nextflow.enable.dsl=2

workflow {
    runBoth( // for adaptive approach
        file('~/Fuzzy_Possibility_Similarity/scripts/AdaptivePossibility_Fuzzy_UMAP.py'),
        // for non adaptive approach
        file('~/Fuzzy_Possibility_Similarity/scripts/Possibility_Fuzzy_UMAP.py')
    )
}

process runBoth {
    input:
    path adaptive
    path noadpative

    output:
    path "*.png"

    container 'python:3.10'

    script:
    """
    pip install matplotlib scikit-learn networkx
    python3 ${adaptive}
    python3 ${noadpative}
    """
}

