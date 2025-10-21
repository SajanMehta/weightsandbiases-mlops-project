# -*- coding: utf-8 -*-
"""
This file is to convert the raw data into version 1 by adding a total lift column
"""

# Import statements

import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

run = wandb.init(project="mlops-datasets", job_type = "create-dataset")

artifact = wandb.Artifact(name="athletes", type="dataset")
artifact.add_file(r"C:\Users\Sajan\OneDrive\Documents\UChicago\MLOps\Assignment_1_second_tool\Data\athletes.csv")

run.log_artifact(artifact, aliases=["raw", "latest"])
run.finish()

run2 = wandb.init(project="mlops-datasets", job_type="load-dataset")
art = run2.use_artifact("smehta15-university-of-chicago/mlops-datasets/athletes:raw")
path = art.download()
data = pd.read_csv(f"{path}/athletes.csv")
run2.finish()
# Data preparation #

# Create total lift
data['total_lift'] = data['snatch'] + data['deadlift'] + data['backsq'] + data['candj']
data = data.dropna(subset=['total_lift'])

# Train Test Split reproducible with random state and constant test size
train, test = train_test_split(data, test_size = 0.2, random_state = 42)

# To csv
train.to_csv(r"C:\Users\Sajan\OneDrive\Documents\UChicago\MLOps\Assignment_1_second_tool\Data\train.csv", index=False)
test.to_csv(r"C:\Users\Sajan\OneDrive\Documents\UChicago\MLOps\Assignment_1_second_tool\Data\test.csv", index = False)
data.to_csv(r"C:\Users\Sajan\OneDrive\Documents\UChicago\MLOps\Assignment_1_second_tool\Data\complete.csv", index = False)

run3 = wandb.init(project="mlops-datasets", job_type="create-dataset")

artifact = wandb.Artifact(name="athletes", type="dataset")
artifact.add_dir(r"C:\Users\Sajan\OneDrive\Documents\UChicago\MLOps\Assignment_1_second_tool\Data")

run3.log_artifact(artifact, aliases=["v1", "latest"])
run3.finish()



