**All steps to clean / transform the data must be documented, with the full source code and/or a detailed log of what was done as appropriate, and the resulting database must be saved as separate files.**

# Fields in policies.csv

| Field label       | Description                                                         | Type of data                  | Range                                             |
|-------------------|---------------------------------------------------------------------|-------------------------------|---------------------------------------------------|
| policy_id         | Unique customer identifier                                          | ID                            | policy_##### (may be shorter than 5 digits)       |
| Quote_dt          | Date the quote was submitted                                        | Approx. continuous            | 2015 - 2018 (yyyy-mm-dd)                          |
| quoted_amt        | Quote amount (US dollars)                                           | Approx. continuous (positive) | $15 - $108,608 or NA                              |
| Prior_carrier_grp | Prior carrier group                                                 | Factor                        | Carrier_1 to 8, Other, NA                         |
| Cov_package_type  | Level of coverage needed                                            | Factor                        | Low, Medium, High, NA                             |
| discount          | Whether or not a discount was applied to the quote amount           | Boolean                       | Yes, No                                           |
| number_drivers    | Number of drivers                                                   | Discrete (positive)           | 1 - 6                                             |
| credit_score      | Credit score of primary policy holder                               | Approx. continuous (positive) | 369 - 850, NA (credit score can go as low as 350) |
| num_loaned_veh    | Number of vehicles on policy that have a loan associated with them  | Discrete (nonnegative)        | 0 - 3                                             |
| num_owned_veh     | Number of owned vehicles on the policy                              | Discrete (nonnegative)        | 1 - 3                                             |
| num_leased_veh    | Number of leased vehicles on the policy                             | Discrete (nonnegative)        | 0 - 2                                             |
| total_number_veh  | Total number vehicles on the policy                                 | Discrete (nonnegative)        | 1 - 8                                             |
| primary_parking   | Where car(s) are primarily parked                                   | Factor                        | home/driveway, parking garage, street, unknown    |
| CAT_zone          | Catastrophe risk zone                                               | Factor                        | 1 - 5, NA                                         |
| Home_policy_ind   | Does customer has existing home insurance policy with Peace of Mind | Boolean                       | Y, N                                              |
| zip               | US zip code of policy holder                                        | Factor                        | #####, NA                                         |
| state_id          | State of policy holder                                              | Factor                        | AL, CT, FL, GA, MN, NJ, NY, WI                    |
| county_name       | County of policy holder                                             | Factor                        | Too numerous to list                              |
| Agent_cd          | Unique agent code (8 digits)                                        | ID                            | ########, NA                                      |
| split             | Train/Test split                                                    | Factor                        | Test, Train                                       |
| convert_ind       | Conversion indicator (0=no, 1=yes). This is the response variable   | Boolean                       | 0, 1 for train data; NA for test data             |

# Fields in drivers.csv

| Field label        | Description                   | Type of data                     | Range                                       |
|--------------------|-------------------------------|----------------------------------|---------------------------------------------|
| policy_id          | Unique customer identifier    | ID                               | policy_##### (may be shorter than 5 digits) |
| gender             | Gender of driver              | Factor / Boolean                 | F, M                                        |
| age                | Age of driver                 | Approx. continuous (nonnegative) | 16 - 147                                    |
| high_education_ind | Higher education indicator    | Boolean                          | 0, 1, NA                                    |
| safty_rating       | Safety rating index of driver | Approx. continuous               | -28 - 100, NA                               |
| living_status      | Driver’s living status        | Factor                           | dependent, own, rent, NA                    |

# Fields in vehicles.csv

| Field label    | Description                                | Type of data           | Range                                            |
|----------------|--------------------------------------------|------------------------|--------------------------------------------------|
| policy_id      | Unique customer identifier                 | ID                     | policy_##### (may be shorter than 5 digits)      |
| car_no         | Unique car identifier (per policy)         | ID                     | 1 - 8                                            |
| ownership_type | Whether the car is loaned, owned or leased | Factor                 | leased, loaned, owned                            |
| color          | Vehicle color                              | Factor                 | black, blue, gray, red, silver, white, other, NA |
| age            | Vehicle age                                | Discrete (nonnegative) | 0 - 16, NA                                       |
| make_model     | Make and model of the vehicle              | Factor                 | Too numerous to list                             |
