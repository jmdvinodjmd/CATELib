# Third-party notices

CATELib's original code is licensed under MIT. Upstream copyright and permission
notices are retained below for the referenced implementation lineage; they are
not replaced by CATELib's copyright notice. Method papers are cited in source
modules. These acknowledgements do not imply endorsement or exact reproduction.

## Single-Cause Perturbation reference implementations

Source: https://github.com/ZhaozhiQIAN/Single-Cause-Perturbation-NeurIPS-2021

Relevant CATELib modules: `catelib/methods/scp.py`, `vsr.py` and
`deconfounder.py`. These are rewritten/adapted implementations of the referenced
baseline constructions. Training, validation and some numerical conventions differ;
see each module's description. The upstream DR-CRN implementation also references
CFR's discrepancy computation, whose notice is retained in the next section.

```text
MIT License

Copyright (c) 2021 Zhaozhi Qian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Counterfactual Regression reference

Source: https://github.com/clinicalml/cfrnet

Relevant CATELib methods: TARNet/CFR and the SCP DR-CFR first-stage variant.
CATELib uses local PyTorch adaptations rather than distributing the upstream
TensorFlow implementation. The upstream reference notice is retained here.

```text
Copyright 2017 Clinical Machine Learning Group

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Separately installed dependencies

NumPy, SciPy, scikit-learn, PyTorch, POT, joblib, pandas and Matplotlib are installed
as dependencies. Their source distributions, binaries, datasets and licence texts
are not vendored into CATELib. Each remains governed by its own licence, including
any bundled third-party components. CATELib's MIT licence does not relicense them.

## Research references versus included code

CISI, CIHSI, NCoRE, TECE and DR are local paper-based implementations, with their
implementation choices documented in the method modules. EconML and CEVAE links
are research/reference links, not bundled copies of those libraries. No CEVAE
source is included in this distribution. Original papers should also be credited
when their methods are used.

