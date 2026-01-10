from pathlib import Path
import pandas as pd

shapes_file = Path("/Users/vignesh/Documents/VoidData/shapes_all_Quijote_0_ss1.0_z0.00_d00.out")
centers_file = Path("/Users/vignesh/Documents/VoidData/centers_all_Quijote_0_ss1.0_z0.00_d00.out")

# need to make something here to iterate through all the simulations
# for simulation in range(2000):
#     shapes_file = Path(f"/Users/vignesh/Documents/VoidData/shapes_all_Quijote_{simulation}_ss1.0_z0.00_d00.out")
#     centers_file = Path(f"/Users/vignesh/Documents/VoidData/centers_all_Quijote_{simulation}_ss1.0_z0.00_d00.out")

# opens the file and reads the data
# skips the header lines
# splits the line into parts
# void_id is the first part
# ellipticity is the second part
# appends the data to the shapes_data list
# creates a dataframe from the shapes_data list
# repeats for the centers_file
# merges the two dataframes on the void_id column
shapes_data = []
with open(shapes_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        void_id = int(parts[0])
        ellipticity = float(parts[1])
        shapes_data.append([void_id, ellipticity])

shapes_df = pd.DataFrame(shapes_data, columns=["void_id", "ellipticity"])

# opens the file and reads the data
# skips the header lines
# splits the line into parts
# radius is the fourth part
# void_id is the seventh part
# appends the data to the centers_data list
# creates a dataframe from the centers_data list
# merges the two dataframes on the void_id column
centers_data = []
with open(centers_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        radius = float(parts[4])
        void_id = int(parts[7])
        density_contrast = float(parts[8])
        centers_data.append([void_id, density_contrast, radius])

# creates a dataframe from the centers_data list
centers_df = pd.DataFrame(centers_data, columns=["void_id", "density_contrast", "radius"])

# merges the two dataframes on the void_id column using an inner join
merged_df = pd.merge(centers_df, shapes_df, on="void_id", how="inner")

# saves the merged dataframe to a csv file
output_csv = Path("/Users/vignesh/Documents/VoidData/simulation_0_voids.csv")
merged_df.to_csv(output_csv, index=False)

# prints the merged dataframe
print(f"Merged CSV saved: {output_csv}")
print(merged_df.head())

'''
using the 0th simulation
example, use radius, density contrast, and ellipticity columns
use numpy to find the min and max
then, use numpy linspace to create 18 bins between min and max, and input the data as well into linspace
generate any distribution plot to see the distribution of voids in each bin
do density as true (which gives the normalized values) or you can do the frequency of voids in each bin
create a single new excel sheet with the three properties and their respective frequencies
it should be like this:
    density contrast | radius | ellipticity
0   0.033             1.345    0.462
1
2
3
4

so at the end, i should i have 2000 excel files

--------------

for the globus part, i can do pip install globus-cli in the vs code terminal
then i need to do globus login in the terminal: $ globus login
then i can do globus transfer SOURCE_ENDPOINT_ID:SOURCE_PATH DESTINATION_ENDPOINT_ID:DESTINATION_PATH --recursive
maybe try it for one center all and one shape all of the same simulation (like 1st simulation) then do all of them

if theres not a way to iteratively download all the files, then i can do it manually for all 2000 files 
by asking chat to make like 4000 lines of code with the globus transfer command, 2000 for centers and 2000 for shapes
look into that more

might need ls *.txt
download the shape and center files for the 0th simulation
do it manually for all 2000 files
need to create the code to iterate through all the simulations
'''