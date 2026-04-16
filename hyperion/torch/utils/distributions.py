"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import re
from typing import Dict, Mapping

import torch
import torch.distributions as dists

PdfDict = Dict[str, torch.Tensor]


def squeeze_pdf(pdf: dists.Distribution, dim: int) -> dists.Distribution:
    """Returns a copy of ``pdf`` with ``dim`` squeezed from distribution tensors."""
    if isinstance(pdf, dists.normal.Normal):
        loc = pdf.loc.squeeze(dim=dim)
        scale = pdf.scale.squeeze(dim=dim)
        return dists.normal.Normal(loc=loc, scale=scale)

    raise NotImplementedError(f"Unsupported distribution type: {type(pdf)}")


def squeeze_pdf_(pdf: dists.Distribution, dim: int) -> None:
    """Squeezes ``dim`` in-place from distribution tensors."""
    if isinstance(pdf, dists.normal.Normal):
        pdf.loc.squeeze_(dim=dim)
        pdf.scale.squeeze_(dim=dim)
        return

    raise NotImplementedError(f"Unsupported distribution type: {type(pdf)}")


def serialize_pdf_to_dict(pdf: dists.Distribution) -> PdfDict:
    """Serializes pdfs to a dictionary

    When we want to return a pdf in a forward function,
    and we are using DataParallel, we need to transform the pdf into a
    dictionary of tensors because DataParallel only is able to combine
    tensors from multiple GPUs but not other objects like distributions.
    """
    if isinstance(pdf, dists.normal.Normal):
        return {"normal.loc": pdf.loc, "normal.scale": pdf.scale}
    else:
        raise NotImplementedError()


def deserialize_pdf_from_dict(pdf: Mapping[str, torch.Tensor]) -> dists.Distribution:
    """Deserializes pdfs from a dictionary.

    When we want to return a pdf in a forward function,
    and we are using DataParallel, we need to transform the pdf into a
    dictionary of tensors because DataParallel only is able to combine
    tensors from multiple GPUs but not other objects like distributions.

    This function will transform the dictionary back into torch.distribution objects
    """
    if len(pdf) == 0:
        raise ValueError("pdf dictionary is empty")

    pdf_type = re.sub(r"\..*$", "", next(iter(pdf)))
    if pdf_type == "normal":
        if "normal.loc" not in pdf or "normal.scale" not in pdf:
            raise KeyError("normal pdf requires keys: 'normal.loc' and 'normal.scale'")
        return dists.normal.Normal(loc=pdf["normal.loc"], scale=pdf["normal.scale"])
    else:
        raise NotImplementedError()
