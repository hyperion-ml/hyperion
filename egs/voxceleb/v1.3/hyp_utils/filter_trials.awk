#!/usr/bin/env awk
# Copyright 2019  Johns Hopkins University (Jesus Villalba) 
# Apache 2.0
BEGIN{
  filter_utts=0;
  filter_models=0;
  if(futts != ""){
    filter_utts=1;
    while(getline < futts)
    {
      utts[$1] = 1
    }
  }
  if(fmodels != ""){
    filter_models=1;
    while(getline < fmodels)
    {
      models[$1] = 1
    }
  }

}
{
  if ((filter_utts==0 || $2 in utts) && (filter_models==0 || $1 in models)){
    print $0
  }
}
