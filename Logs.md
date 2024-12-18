Logs:

* Dec 17th, ran inference to obtain embeddings of ~800k sequences. Split dataset into four chunks due to memory issues. Ran each individually with its on slurm and python script.
12 hours was not enough to embed 200k sequences, changed job limit to 48 hours.