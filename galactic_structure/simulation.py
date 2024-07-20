import numpy as np
from starcat import IMF
from astropy.table import Table
import joblib

# typical number of stars in a field
TYPICAL_NUMBER_OF_STARS = 1000

# typical contamination rate
MAX_CONTAMINATION_RATE = 0.2

# [M/H]
MIN_FIELD_MH = -2.0
MAX_FIELD_MH = 0.5
MIN_POPULATION_MH = -2.0
MAX_POPULATION_MH = 0.5

# AGE
# np.log10(137e8) = 10.136720567156408
MIN_FIELD_LOGAGE = 6
MAX_FIELD_LOGAGE = 10.13
MIN_POPULATION_LOGAGE = 6
MAX_POPULATION_LOGAGE = 10.13


# IMF
imf = IMF()

# draw Poisson random numbers
n1 = np.random.poisson(TYPICAL_NUMBER_OF_STARS)
n2 = int(np.random.uniform(low=0.0, high=MAX_CONTAMINATION_RATE) * n1)

# determine LOGAGE and MH
age = np.random.uniform(
    low=MIN_POPULATION_LOGAGE,
    high=MAX_POPULATION_LOGAGE,
)
mh = np.random.uniform(low=MIN_POPULATION_MH, high=MAX_POPULATION_MH)


# sample
mass_sample = np.hstack(
    (
        imf.sample(n_stars=n1, mass_min=0.09, mass_max=100),
        imf.sample(n_stars=n2, mass_min=0.09, mass_max=100),
    )
)

age_sample = np.hstack(
    (
        np.ones(n1, dtype=float) * age,
        np.ones(n2, dtype=float)
        * np.random.uniform(
            low=MIN_FIELD_LOGAGE,
            high=MAX_FIELD_LOGAGE,
            size=n2,
        ),
    )
)

mh_sample = np.hstack(
    (
        np.ones(n1, dtype=float) * mh,
        np.ones(n2, dtype=float)
        * np.random.uniform(low=MIN_FIELD_MH, high=MAX_FIELD_MH, size=n2),
    )
)

label_sample = np.hstack(
    (
        np.ones(n1, dtype=int),
        np.zeros(n2, dtype=int),
    )
)

sample_table = Table(
    data=[mass_sample, age_sample, mh_sample, label_sample],
    names=["mass_sample", "age_sample", "mh_sample", "label_sample"],
)

meta = dict(
    n1=n1,
    n2=n2,
    age=age,
    mh=mh,
)
sample_table.meta.update(**meta)

sample_table.write("test_sample.fits", overwrite=True)
sample_table.show_in_browser()
print(sample_table.meta)
