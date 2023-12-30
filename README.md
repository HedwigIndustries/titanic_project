## engineering-practices-project-titanic

#### Short description: Predicting the survival of Titanic passengers using machine learning algorithms.

#### The project has a `snakemake` pipeline configured.

#### The project supports `git lfs` interment.

You can find rules that can execute the pipeline, which located in the file `run.smk`.

#### To run some task use:

```bash
snakemake  -s run.smk --cores all -R {task_name}  
```

#### After this commands, you can see result of all pipeline:

```bash
snakemake  -s run.smk --cores all -R prepare_data  
snakemake  -s run.smk --cores all -R train_model  
snakemake  -s run.smk --cores all -R predict_params  
```

Data analysis, graphs, data validation stages, you will find all this in this report `data/titanic.ipynb`.
