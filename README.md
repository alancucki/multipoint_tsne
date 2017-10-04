# Multipoint t-SNE

This is a Python implementation of multipoint extension described
in the paper [Multipoint Nebighbor Embedding](https://link.springer.com/chapter/10.1007/978-3-319-64206-2_51),
based on [scikit-learn](http://scikit-learn.org/stable/).
It enables embedding individual datapoints as sets of datapoints,
useful in portraying datapoints of multiple relationships,
e.g., polysemic words, pictures of multiple classes, etc.

As this is experimental code, feel free to submit features and bugfixes.

## Requirements

* Theano
* scikit-learn

## Usage

Syntax is similar to that of [scikit-learn TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). Copies of points are introduced in stages. For this reason, the crucial thing is the *schedule* expressed as a list of tuples. Each tuple holds `(<exaggerate>, <num-iterations>, <copy-percentage>)`. For instance, vanilla t-SNE schedule is `[(True, 250, 0.0), (False, 750, 0.0)]`.

A sample run on dataset `x`:
```                   
mtsne = MultipointTSNE(
    n_components=2, perplexity=50.0, early_exaggeration=4.0,
    learning_rate=3.5, optimizer='adam', initial_dims=50,
    train_schedule=[
       (True,  250, 0.0),
       (False, 200, 0.0),
       (False, 200, 5.0),
       (False, 200, 5.0),
       (False, 200, 5.0),
       (False, 200, 5.0),
       (False, 200, 5.0),
       (False, 200, 5.0),
       (False, 200, 5.0),
       (False, 500, 0.0),
    ],
    verbose=15, n_iter_check=100, random_state=1234,
    method='barnes_hut', angle=0.5, cleanup_thresh=0.15)
y = mtsne.run(x=x)
```

Due to the dynamic nature of the algorithm, it might be resumed:
```               
mtsne.run_schedule([(False, 200, 10.0), (False, 500, 0.0)])
```

## Known Bugs
During phases with many iterations, unnecessary copies
get pushed infinitely far from the main cloud of points
causing numerical errors.

Citation
========
```
@Inbook{Lancucki2017,
  author="Lancucki, Adrian and Chorowski, Jan",
  editor="Ek{\v{s}}tein, Kamil and Matou{\v{s}}ek, V{\'a}clav",
  title="Multipoint Neighbor Embedding",
  bookTitle="Text, Speech, and Dialogue: 20th International Conference, TSD 2017, Prague, Czech Republic, August 27-31, 2017, Proceedings",
  year="2017",
  publisher="Springer International Publishing",
  address="Cham",
  pages="456--464",
  isbn="978-3-319-64206-2",
  doi="10.1007/978-3-319-64206-2_51",
  url="https://doi.org/10.1007/978-3-319-64206-2_51"
}
```
