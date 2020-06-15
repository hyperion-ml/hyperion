"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import re

import torch
import torch.distributions as dists

def serialize_pdf_to_dict(pdf):
    """Serializes pdfs to a dictionary
    
    When we want to return a pdf in a forward function, 
    and we are using DataParallel, we need to transform the pdf into a
    dictionary of tensors because DataParallel only is able to combine 
    tensors from multiple GPUs but not other objects like distributions.
    """
    if isinstance(pdf, dists.normal.Normal):
        return {'normal.loc': pdf.loc, 'normal.scale': pdf.scale }
    else:
        raise NotImplementedError()



def deserialize_pdf_from_dict(pdf):
    """Derializes pdfs from a dictionary
    
    When we want to return a pdf in a forward function, 
    and we are using DataParallel, we need to transform the pdf into a
    dictionary of tensors because DataParallel only is able to combine 
    tensors from multiple GPUs but not other objects like distributions.

    This function will transform the dictionary back into torch.distribution objects
    """
    pdf_type = re.sub(r'.*','', pdf.keys()[0])
    if pdf_type == 'normal':
        return dists.normal.Normal(
            loc=pdf['normal.loc'], scale=pdf['normal.scale'])
    else:
        raise NotImplementedError()
    
    


