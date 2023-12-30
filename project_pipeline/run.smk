# Snakefile to run pipeline

rule prepare_data:
    input:
        train_data = "../data/train.csv",
    output:
        prepared_data = "../data/prepared_data.csv",
    shell:
        """
        python3 prepare_data.py {input.train_data} {output.prepared_data}
        """

rule train_model:
    input:
        prepared_data = "../data/prepared_data.csv"
    output:
        models_and_data = "../data/models_and_data.pkl"
    shell:
        """
        python3 train_model.py {input.prepared_data} {output.models_and_data}
        """

rule predict_params:
    input:
        models_and_data = "../data/models_and_data.pkl",
    output:
        predictions = "../data/predictions.txt"
    shell:
        """
        python3 predict_params.py {input.models_and_data} {output.predictions}
        """