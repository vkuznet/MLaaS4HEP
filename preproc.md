### Performing preprocessing operations using a preproc.json file

The MLaaS4HEP code has been updated to support [uproot4](https://uproot.readthedocs.io/en/latest/basic.html) and to allow users to perform pre-processing operations on the input ROOT data.
The migration to the updated version of uproot allowed to create new branches and to apply cuts, both on new and on existing branches. In MLaaS4HEP now this is possible providing a `preproc.json` file as argument while running the `workflow.py` script:

```
workflow --files=files.txt --labels=labels.txt --model=ex_keras.py --preproc=preproc.json
```

### Structure of preproc.json

A simple example of a `preproc.json` file is the following:

```
{
  "new_branch": {
    "log_Energy": {
      "def": "log(Energy)",
      "type": "jagged",
      "cut_1": ["log_Energy<6.31", "any"],
      "cut_2": ["log_Energy>5.85", "all"],
      "remove": "False",
      "keys_to_remove": ["Energy"]
    },

    "nMuon_square": {
      "def": "nMuon**2",
      "type": "flat",
      "cut": "1<=nMuon_square<=16",
      "remove": "False",
      "keys_to_remove": ["nMuon"]
    }
  },
  "flat_cut": {
    "nLeptons": {
      "cut": "nLeptons<=2",
      "remove": "False"
    }
  },
  "jagged_cut": {
    "Muon_Pt": {
      "cut": ["Muon_Pt>200", "all"],
      "remove": "False"
    }
  }
}

```

This file is structured in 3 main categories:
- `new_branch`,
- `flat_cut`,
- `jagged_cut`.

The `new_branch` component allows the user to define new branches and gives the possibility to apply cuts on them, while `flat_cut` and `jagged_cut` allow to apply cuts on existing `flat` and `jagged` branches respectively. 
Basically `flat` branches store simple values (e.g. float or integer numbers), while not-flat branches also called `jagged` contain vectors which dimension can vary along the branch. For a more detailed description on this matter, please read [this](https://link.springer.com/article/10.1007/s41781-021-00061-3). If the user doesn't want to create new branches or apply cuts on `flat` or `jagged branches`, just remove the corresponding category from the `preproc.json` file.

### new_branch

A new branch can be defined through mathematical operations involving existing branches. 
To create new branches the user should provide under the `new_branch` category a series of information:
- the name of the new branch,
- the mathematical definition of the new branch involving the existing branches, which supports both the basic operations (+, -, *, /, **) and the numpy functions listed [here](https://github.com/scikit-hep/uproot4/blob/916085ae24c382404254756c86afe760acdece56/src/uproot/language/python.py#L237),
- the type of the new branch (`flat` or `jagged`),
- the cut to apply,
- remove or not the new branch after the cut,
- the list of branches to be removed after creation of the new branch.

In case of cuts on `jagged` branches the user should specify the type of the cut, choosing between `all` and `any`.
The [all](https://awkward-array.readthedocs.io/en/latest/_auto/ak.all.html) type should be used when the cut condition must be satisfied by all the values of a given `jagged` branch, while the [any](https://awkward-array.readthedocs.io/en/latest/_auto/ak.any.html) type if at least one element in a given `jagged` branch satisfies the cut condition.
In case of multiple cuts, they must be orderded using `cut_i` as key, where `i` indicates the number of the cut. If the user does not want to apply cuts on the new branch or doesn't want to remove any branch, then the `cut` and the `keys_to_remove` must be removed respectively from the `preproc.json` file.


### flat_cut and jagged_cut
To apply cuts on `flat` and `jagged` branches the structure to use is similar to the previous one. The following information must be provided:
- the name of the branch to cut on,
- the cut to apply,
- remove or not the branch after the cut.

To apply more than one cut within the same branch just number the `cut` key with `cut_i` as seen in the previous example. 
