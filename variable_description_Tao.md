My cleaned data is stored in file ``tao.RData``, which after loading you could find train and test datasets.

Both of the two datasets have complete variables including those from drivers and vehicles.

I gave a brief description of how I transfer different variables to ready-to-use ones.

Policies.csv
policy_id : Not change.
quote_dt : Not change.
quoted_amt: To numeric variable.
prior_carrier_grp: To 8 dummy variables: Carrier1 - Carrier8.
cov_package_type: To 3 dummy variables: Package_high, Package_medium, Package_low.
discount: To a dummy varialb.
number_drivers: Not change.
credit_score: Not change.
num_loaned_veh: Not change.
num_owned_veh: Not change.
num_leased_veh: Not change.
total_number_veh: Not change.
primary_parking: To 3 dummy variables: Parking_street, Parking_home, Parking_garage (I did not code type of 'unknown').
CAT_zone: To 5 dummy variables: Catzone1 - Catzone5.
home_policy_ind: To dummy variable.
zip: Not change.
state_id: To dummy varialbes with the name as the state.
county_name: Not change.
agent_cd: Not change.
split: Train/Test split
convert_ind: Conversion indicator (0=no, 1=yes). This is the response variable

Drivers.csv:
policy_id: Not change.
gender: Code male as 1, female as 0, then use the mean value as mean_gender.
age: Compute the mean value as mean_age.
high_education_ind: Compute the mean value as mean_edu (na.rm = T).
safty_rating: Compute the mean value as mean_safe (na.rm = T).
living_status: Compute the number for each status as num_liv_own, num_liv_dep, num_liv_rent, also compute the total number of drivers for each policy as act_num_drivers.

Vehicles.csv:
policy_id: Not change.
car_no: Not change.
ownership_type: Compute the number for each type as act_num_leased_veh, act_num_loaned_veh, act_num_owned_veh, also compute the total number of drivers for each policy as act_num_veh.
color: Not change.
age: Compute the mean value as mean_car_age (na.rm = T).
make_model: To dummy varialbes with the name as the brand.