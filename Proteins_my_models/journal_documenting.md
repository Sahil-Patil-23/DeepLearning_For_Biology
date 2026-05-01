In this text document. I will document the what/why/how I did things. 


February 17, 2026
-----------------------------
* Since I'm using Mac Silicon, I can't use `torch.cuda`; using CPU also doesn't make sense. What is available to me is
`Metal Performance Shaders`- and Pytorch has built-in support for it. I have included a cell in the `protein_ML.ipynb` that checks whether my notebook can access my Mac's GPU. 
* I also installed versions of PyTorch and Jax using pip.
* `uv pip compile requirements/base.txt requirements/dlfb.txt requirements/proteins.txt \
  --constraint requirements/constraints.txt | \
  uv pip install -r -`. Used this command instead of the one listed in the O'Reilly Python notebook.
* To get the actual data used in Chapter 2 of the O'Reilly book, I had to iron out some more issues. Since the O'Reilly notebook is made to be used in Google Colab & not on local hard drives, I had to create a local data/ folder & run the command `dlfb-provision --chapter proteins --no-models --destination ./data` to get the relevant data into my local system.
* Gemini has been very helpful in getting through logistic hell