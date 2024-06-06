import os
import pandas as pd
import numpy as np

USE_MINI = True

def create_csv():
    inat_path = f"/mnt/hdd-4t/datasets/inaturalist{'' if USE_MINI else '_full'}"
    train_path = os.path.join(inat_path, "2021_train_mini" if USE_MINI else "2021_train")
    val_path = os.path.join(inat_path, "2021_valid")

    df = pd.DataFrame(columns=["id", "kingdom", "phylum", "class", "order", "family", "genus", "species", "n_train_samples", "n_val_samples"])

    for filename in os.listdir(val_path):
        id, kingdom, phylum, class_, order, family, genus, species = filename.split("_")
        n_val_samples = len(os.listdir(os.path.join(val_path, filename)))
        n_train_samples = len(os.listdir(os.path.join(train_path, filename)))
        df = df._append({"id": id, "kingdom": kingdom, "phylum": phylum, "class": class_, "order": order, "family": family, "genus": genus, "species": species, "n_train_samples": n_train_samples, "n_val_samples": n_val_samples}, ignore_index=True)

    df.to_csv(f"inat_taxonomy{'_mini' if USE_MINI else ''}.csv", index=False)

#create_csv()

df = pd.read_csv(f"inat_taxonomy{'_mini' if USE_MINI else ''}.csv")

df.kingdom.unique()

print(len(df.family.unique()))

print(df[(df.kingdom == "Animalia")].groupby("phylum").size().sort_values(ascending=False))

selected_phylum = ['Chordata']
genus_subset = df[(df.kingdom == "Animalia") & (df.phylum.isin(selected_phylum))]
unique_family = genus_subset.groupby("family").size().sort_values(ascending=False).index
print(len(unique_family))
genus_subset = genus_subset[genus_subset.family.isin(unique_family[:265])]
print(len(genus_subset.genus.unique()))

genus_ids = genus_subset.drop_duplicates(subset=["genus"])["id"]  # only keep one entry per genus
print(genus_subset[genus_subset.id.isin(genus_ids)].n_train_samples.sum())
np.save("datasets/inat_genus_ids.npy", genus_ids.values)

print("===")
#print(genus_subset.groupby("genus").size().sort_values(ascending=False))
unique_genus = genus_subset.groupby("genus").size().sort_values(ascending=False).index
species_subset = genus_subset[genus_subset.genus.isin(unique_genus[:277])]
species_subset.drop_duplicates(subset=["species"] , inplace=True)  # only keep one entry per species
print(len(species_subset.species.unique()))

species_ids = species_subset.drop_duplicates(subset=["species"])["id"]  # only keep one entry per genus
print(species_subset[species_subset.id.isin(species_ids)].n_train_samples.sum())
np.save("datasets/inat_species_ids.npy", species_ids.values)
